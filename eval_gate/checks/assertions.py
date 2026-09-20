"""Assertion checks over the model answer text.

Implements: ``must_contain``, ``must_not_contain``, ``regex``, ``json_schema``
(structural validity of JSON output), and ``refusal`` (did the model refuse /
abstain when it should, or answer when it should).
"""

from __future__ import annotations

import json
import re
from typing import Any

from eval_gate.checks import (
    CheckContext,
    CheckOutcome,
    _bool_flag,
    _non_empty_str,
    _param,
    register,
)
from eval_gate.config import CheckSpec


class SchemaDefinitionError(ValueError):
    """Raised when a ``json_schema`` check uses a keyword this validator cannot enforce.

    Silently ignoring an unsupported keyword is worse than not supporting it: a
    suite author writes ``additionalProperties: false`` or ``pattern``, reads a
    green check, and believes a contract is being enforced that is not
    (2026-09-19 audit, finding D7).
    """

# Phrases that signal a model declining / abstaining. Deliberately narrow to
# avoid false positives on normal answers.
_REFUSAL_PATTERNS = [
    r"\bi (?:can(?:'|no)?t|cannot|won'?t|am (?:un)?able to)\b",
    r"\bi (?:do not|don'?t) have (?:that|the|any|enough) (?:information|data|context)\b",
    r"\bi'?m (?:sorry|unable)\b",
    r"\bno information (?:is )?available\b",
    r"\bnot (?:covered|found|mentioned|specified) in the (?:context|documents?|policy)\b",
    r"\bi'?m not able to (?:help|assist|answer)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    """Heuristically detect whether *text* is a refusal / abstention."""
    return bool(_REFUSAL_RE.search(text))


def _value_and_flag(spec: CheckSpec) -> list[str]:
    return _non_empty_str(spec, "value") + _bool_flag(spec, "case_insensitive")


@register(
    "must_contain",
    params={"value", "case_insensitive"},
    required={"value"},
    validator=_value_and_flag,
)
def _must_contain(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    value = str(_param(spec, "value"))
    ci = bool(_param(spec, "case_insensitive", required=False, default=False))
    hay = ctx.answer.lower() if ci else ctx.answer
    needle = value.lower() if ci else value
    ok = needle in hay
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=(f"answer contains {value!r}" if ok else f"answer is missing {value!r}"),
        detail={"value": value, "case_insensitive": ci},
    )


@register(
    "must_not_contain",
    params={"value", "case_insensitive"},
    required={"value"},
    validator=_value_and_flag,
)
def _must_not_contain(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    value = str(_param(spec, "value"))
    ci = bool(_param(spec, "case_insensitive", required=False, default=False))
    hay = ctx.answer.lower() if ci else ctx.answer
    needle = value.lower() if ci else value
    ok = needle not in hay
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=(f"answer avoids {value!r}" if ok else f"answer unexpectedly contains {value!r}"),
        detail={"value": value, "case_insensitive": ci},
    )


def _compilable_pattern(spec: CheckSpec) -> list[str]:
    errors = _non_empty_str(spec, "pattern") + _bool_flag(spec, "case_insensitive")
    if errors:
        return errors
    try:
        re.compile(str(spec.params["pattern"]))
    except re.error as exc:
        return [f"check 'regex': 'pattern' does not compile: {exc}"]
    return []


@register(
    "regex",
    params={"pattern", "case_insensitive"},
    required={"pattern"},
    validator=_compilable_pattern,
)
def _regex(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    pattern = str(_param(spec, "pattern"))
    ci = bool(_param(spec, "case_insensitive", required=False, default=False))
    flags = re.IGNORECASE if ci else 0
    ok = bool(re.search(pattern, ctx.answer, flags))
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=(f"answer matches /{pattern}/" if ok else f"answer does not match /{pattern}/"),
        detail={"pattern": pattern},
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json(text: str) -> Any:
    """Parse JSON from *text*, tolerating a single ```json fenced block."""
    candidate = text.strip()
    candidate = _FENCE_RE.sub("", candidate).strip()
    return json.loads(candidate)


SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "type",
        "required",
        "properties",
        "items",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        # Annotations that carry no constraint, so ignoring them enforces nothing
        # that was promised.
        "title",
        "description",
    }
)

_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)


def validate_schema_definition(schema: Any, path: str = "$") -> list[str]:
    """Return every reason *schema* cannot be enforced by :func:`validate_schema`.

    Used by the suite loader so an unenforceable schema is a config error at
    load time rather than a check that quietly enforces less than it says.
    """
    if not isinstance(schema, dict):
        return [f"{path}: schema must be a mapping, got {type(schema).__name__}"]

    errors: list[str] = []
    unsupported = sorted(set(schema) - SUPPORTED_SCHEMA_KEYWORDS)
    if unsupported:
        errors.append(
            f"{path}: unsupported schema keyword(s) {unsupported}; this validator "
            f"enforces only {sorted(SUPPORTED_SCHEMA_KEYWORDS)}"
        )
    declared_type = schema.get("type")
    if declared_type is not None and declared_type not in _SCHEMA_TYPES:
        errors.append(f"{path}: unknown schema type '{declared_type}'")
    props = schema.get("properties")
    if props is not None:
        if not isinstance(props, dict):
            errors.append(f"{path}.properties: must be a mapping")
        else:
            for key, sub in props.items():
                errors.extend(validate_schema_definition(sub, f"{path}.{key}"))
    if "items" in schema:
        errors.extend(validate_schema_definition(schema["items"], f"{path}[]"))
    if "required" in schema and not isinstance(schema["required"], list):
        errors.append(f"{path}.required: must be a list of property names")
    return errors


def validate_schema(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate *instance* against a JSON-Schema subset. Returns a list of errors.

    Supported keywords: ``type`` (object/array/string/number/integer/boolean/null),
    ``required``, ``properties``, ``items``, ``enum``, ``minimum``, ``maximum``,
    ``minLength``, ``maxLength``. Enough to assert real output contracts without
    pulling in a heavyweight dependency.

    Raises:
        SchemaDefinitionError: If the schema uses a keyword this validator
            cannot enforce. It refuses rather than ignoring it, so a schema
            never enforces less than it appears to.
    """
    definition_errors = validate_schema_definition(schema, path)
    if definition_errors:
        raise SchemaDefinitionError("; ".join(definition_errors))

    errors: list[str] = []
    _TYPE_CHECKS = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }

    expected_type = schema.get("type")
    if expected_type is not None:
        checker = _TYPE_CHECKS.get(expected_type)
        if checker is None:
            errors.append(f"{path}: unknown schema type '{expected_type}'")
            return errors
        if not checker(instance):
            errors.append(f"{path}: expected type '{expected_type}', got {type(instance).__name__}")
            return errors  # further checks assume the type held

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in enum {schema['enum']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: length {len(instance)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: length {len(instance)} > maxLength {schema['maxLength']}")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property '{key}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(validate_schema(instance[key], subschema, f"{path}.{key}"))

    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))

    return errors


def _schema_param(spec: CheckSpec) -> list[str]:
    schema = spec.params.get("schema")
    if not isinstance(schema, dict) or not schema:
        return [f"check 'json_schema': 'schema' must be a non-empty mapping, got {schema!r}"]
    return validate_schema_definition(schema)


@register("json_schema", params={"schema"}, required={"schema"}, validator=_schema_param)
def _json_schema(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    schema = _param(spec, "schema")
    if not isinstance(schema, dict):
        raise KeyError("check 'json_schema' requires a mapping 'schema' parameter")
    try:
        instance = _extract_json(ctx.answer)
    except json.JSONDecodeError as exc:
        return CheckOutcome(
            check_type=spec.type,
            passed=False,
            message=f"answer is not valid JSON: {exc}",
            detail={"errors": [str(exc)]},
        )
    errors = validate_schema(instance, schema)
    ok = not errors
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=("answer is valid JSON matching schema" if ok else f"{len(errors)} schema error(s)"),
        detail={"errors": errors},
    )


@register(
    "refusal",
    params={"expected"},
    validator=lambda spec: _bool_flag(spec, "expected"),
)
def _refusal(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    expected = bool(_param(spec, "expected", required=False, default=True))
    got = is_refusal(ctx.answer)
    ok = got == expected
    if expected:
        msg = "answer correctly refused/abstained" if ok else "answer should have refused but did not"
    else:
        msg = "answer correctly did not refuse" if ok else "answer unexpectedly refused/abstained"
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=msg,
        detail={"expected_refusal": expected, "detected_refusal": got},
    )
