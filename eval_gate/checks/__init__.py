"""Check framework: shared types, a registry, and dispatch.

A *check* is a small piece of real logic that inspects a case's model answer
and/or retrieval result and returns a pass/fail outcome with a message. Checks
are registered by name and referenced from the suite YAML by ``type``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from eval_gate.config import CheckSpec
from eval_gate.corpus import Corpus


@dataclass(frozen=True)
class CheckContext:
    """Everything a check needs to evaluate one case.

    Attributes:
        answer: The model's answer text for the case.
        retrieved_ids: Ordered document ids returned by the retriever.
        relevant_ids: Ground-truth relevant document ids for the case.
        corpus: The document corpus (for grounding / faithfulness checks).
    """

    answer: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    corpus: Corpus


@dataclass(frozen=True)
class CheckOutcome:
    """The result of running one check.

    Attributes:
        check_type: The check's registered type name.
        passed: Whether the check passed.
        message: Human-readable explanation (always populated).
        detail: Optional structured detail (metric values, etc.).
    """

    check_type: str
    passed: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


CheckFn = Callable[[CheckSpec, CheckContext], CheckOutcome]
ValidatorFn = Callable[[CheckSpec], list[str]]


@dataclass(frozen=True)
class CheckDecl:
    """A registered check: its implementation and its parameter contract.

    The contract is declared on the function itself so a check cannot be added
    without saying which parameters it accepts. Before the 2026-09-19 audit
    there was no contract at all: a misspelled parameter was silently ignored
    and the check ran with its default, which is how ``length`` with a typo'd
    ``max_word`` became a check that could not fail.

    Attributes:
        fn: The check implementation.
        params: Every parameter name the check accepts.
        required: Parameters that must be present.
        validator: Optional extra validation returning a list of error strings.
    """

    fn: CheckFn
    params: frozenset[str]
    required: frozenset[str]
    validator: ValidatorFn | None = None


_REGISTRY: dict[str, CheckDecl] = {}


def register(
    name: str,
    *,
    params: set[str] | frozenset[str] = frozenset(),
    required: set[str] | frozenset[str] = frozenset(),
    validator: ValidatorFn | None = None,
) -> Callable[[CheckFn], CheckFn]:
    """Decorator to register a check function under *name* with its param contract."""
    unknown_required = set(required) - set(params)
    if unknown_required:
        raise ValueError(f"check '{name}': required params not declared: {sorted(unknown_required)}")

    def deco(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            raise ValueError(f"check '{name}' already registered")
        _REGISTRY[name] = CheckDecl(
            fn=fn,
            params=frozenset(params),
            required=frozenset(required),
            validator=validator,
        )
        return fn

    return deco


def registered_checks() -> list[str]:
    """Sorted names of all registered checks."""
    return sorted(_REGISTRY)


def declared_params(name: str) -> frozenset[str] | None:
    """Parameters declared by check *name*, or ``None`` if it is not registered."""
    decl = _REGISTRY.get(name)
    return None if decl is None else decl.params


def run_check(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    """Dispatch and run a single check.

    Raises:
        KeyError: If ``spec.type`` is not a registered check.
    """
    decl = _REGISTRY.get(spec.type)
    if decl is None:
        raise KeyError(
            f"unknown check type '{spec.type}'. Registered: {registered_checks()}"
        )
    return decl.fn(spec, ctx)


def validate_check_spec(spec: CheckSpec) -> list[str]:
    """Return every reason *spec* cannot be run as written (empty list = usable).

    This is what stops a check that cannot fail from reaching a run: an unknown
    check type, a misspelled or unknown parameter, a missing required
    parameter, or a parameter value that makes the check vacuous (an empty
    ``must_contain`` value, a ``length`` with no bounds, and so on).
    """
    decl = _REGISTRY.get(spec.type)
    if decl is None:
        return [f"unknown check type '{spec.type}'. Registered: {registered_checks()}"]

    errors: list[str] = []
    unknown = sorted(set(spec.params) - decl.params)
    if unknown:
        errors.append(
            f"unknown parameter(s) {unknown} for check '{spec.type}'; "
            f"accepted: {sorted(decl.params)}"
        )
    missing = sorted(decl.required - set(spec.params))
    if missing:
        errors.append(f"check '{spec.type}' requires parameter(s) {missing}")
    if not errors and decl.validator is not None:
        errors.extend(decl.validator(spec))
    return errors


# --- shared parameter validators -------------------------------------------
# Each returns a list of error strings so several problems can be reported at
# once instead of one per run.


def _non_empty_str(spec: CheckSpec, key: str) -> list[str]:
    value = spec.params.get(key)
    if not isinstance(value, str) or not value.strip():
        return [
            f"check '{spec.type}': '{key}' must be a non-empty string "
            f"(got {value!r}); an empty value matches everything and cannot fail"
        ]
    return []


def _bool_flag(spec: CheckSpec, key: str) -> list[str]:
    if key in spec.params and not isinstance(spec.params[key], bool):
        return [f"check '{spec.type}': '{key}' must be true or false, got {spec.params[key]!r}"]
    return []


def _unit_interval(spec: CheckSpec, key: str) -> list[str]:
    if key not in spec.params:
        return []
    value = spec.params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"check '{spec.type}': '{key}' must be a number in [0, 1], got {value!r}"]
    if not 0.0 <= float(value) <= 1.0:
        return [f"check '{spec.type}': '{key}' must be in [0, 1], got {value!r}"]
    return []


def _positive_int(spec: CheckSpec, key: str) -> list[str]:
    if key not in spec.params:
        return []
    value = spec.params[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return [f"check '{spec.type}': '{key}' must be an integer >= 1, got {value!r}"]
    return []


def _param(spec: CheckSpec, key: str, required: bool = True, default: Any = None) -> Any:
    if key in spec.params:
        return spec.params[key]
    if required:
        raise KeyError(f"check '{spec.type}' requires parameter '{key}'")
    return default


# Import submodules so their @register decorators run. Kept at the bottom to
# avoid circular imports (submodules import from this module).
from eval_gate.checks import assertions as _assertions  # noqa: E402,F401
from eval_gate.checks import quality as _quality  # noqa: E402,F401
from eval_gate.checks import retrieval as _retrieval  # noqa: E402,F401
