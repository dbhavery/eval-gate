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

_REGISTRY: dict[str, CheckFn] = {}


def register(name: str) -> Callable[[CheckFn], CheckFn]:
    """Decorator to register a check function under *name*."""

    def deco(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            raise ValueError(f"check '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def registered_checks() -> list[str]:
    """Sorted names of all registered checks."""
    return sorted(_REGISTRY)


def run_check(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    """Dispatch and run a single check.

    Raises:
        KeyError: If ``spec.type`` is not a registered check.
    """
    fn = _REGISTRY.get(spec.type)
    if fn is None:
        raise KeyError(
            f"unknown check type '{spec.type}'. Registered: {registered_checks()}"
        )
    return fn(spec, ctx)


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
