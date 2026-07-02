"""Assertion checks over the model answer text.

Implements: ``must_contain``, ``must_not_contain``, ``regex``, ``json_schema``
(structural validity of JSON output), and ``refusal`` (did the model refuse /
abstain when it should, or answer when it should).
"""

from __future__ import annotations

import json
import re
from typing import Any

from eval_gate.checks import CheckContext, CheckOutcome, _param, register
from eval_gate.config import CheckSpec

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


@register("must_contain")
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


@register("must_not_contain")
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


@register("regex")
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


def validate_schema(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate *instance* against a JSON-Schema subset. Returns a list of errors.

    Supported keywords: ``type`` (object/array/string/number/integer/boolean/null),
    ``required``, ``properties``, ``items``, ``enum``, ``minimum``, ``maximum``,
    ``minLength``, ``maxLength``. Enough to assert real output contracts without
    pulling in a heavyweight dependency; unsupported keywords are ignored.
    """
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


@register("json_schema")
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


@register("refusal")
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
