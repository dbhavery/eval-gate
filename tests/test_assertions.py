"""Unit tests for assertion checks and the JSON-schema validator."""

from __future__ import annotations

from eval_gate.checks import CheckContext, run_check
from eval_gate.checks.assertions import is_refusal, validate_schema
from eval_gate.config import CheckSpec


def _ctx(answer: str, corpus) -> CheckContext:
    return CheckContext(answer=answer, retrieved_ids=[], relevant_ids=[], corpus=corpus)


def test_must_contain_pass_and_fail(tiny_corpus):
    ctx = _ctx("Refunds within 30 days.", tiny_corpus)
    assert run_check(CheckSpec("must_contain", {"value": "30 days"}), ctx).passed
    assert not run_check(CheckSpec("must_contain", {"value": "60 days"}), ctx).passed


def test_must_contain_case_insensitive(tiny_corpus):
    ctx = _ctx("REFUND policy applies.", tiny_corpus)
    assert run_check(
        CheckSpec("must_contain", {"value": "refund", "case_insensitive": True}), ctx
    ).passed
    assert not run_check(CheckSpec("must_contain", {"value": "refund"}), ctx).passed


def test_must_not_contain(tiny_corpus):
    ctx = _ctx("A clean, grounded answer.", tiny_corpus)
    assert run_check(CheckSpec("must_not_contain", {"value": "hallucination"}), ctx).passed
    assert not run_check(CheckSpec("must_not_contain", {"value": "clean"}), ctx).passed


def test_regex(tiny_corpus):
    ctx = _ctx("Order #12345 shipped.", tiny_corpus)
    assert run_check(CheckSpec("regex", {"pattern": r"#\d{5}"}), ctx).passed
    assert not run_check(CheckSpec("regex", {"pattern": r"#\d{8}"}), ctx).passed


def test_json_schema_valid(tiny_corpus):
    ctx = _ctx('{"policy": "refund", "days": 30}', tiny_corpus)
    spec = CheckSpec(
        "json_schema",
        {"schema": {"type": "object", "required": ["policy", "days"],
                    "properties": {"policy": {"type": "string"}, "days": {"type": "integer"}}}},
    )
    assert run_check(spec, ctx).passed


def test_json_schema_invalid_type_and_missing(tiny_corpus):
    ctx = _ctx('{"policy": "refund", "days": "thirty"}', tiny_corpus)
    spec = CheckSpec(
        "json_schema",
        {"schema": {"type": "object", "required": ["policy", "days", "id"],
                    "properties": {"days": {"type": "integer"}}}},
    )
    out = run_check(spec, ctx)
    assert not out.passed
    # both a type error and a missing-required error should be reported
    assert any("integer" in e for e in out.detail["errors"])
    assert any("id" in e for e in out.detail["errors"])


def test_json_schema_not_json(tiny_corpus):
    ctx = _ctx("this is plain prose, not json", tiny_corpus)
    spec = CheckSpec("json_schema", {"schema": {"type": "object"}})
    assert not run_check(spec, ctx).passed


def test_json_schema_tolerates_code_fence(tiny_corpus):
    ctx = _ctx('```json\n{"days": 30}\n```', tiny_corpus)
    spec = CheckSpec("json_schema", {"schema": {"type": "object", "properties": {"days": {"type": "integer"}}}})
    assert run_check(spec, ctx).passed


def test_validate_schema_enum_and_bounds():
    schema = {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["open", "closed"]},
        "count": {"type": "integer", "minimum": 0, "maximum": 10},
    }}
    assert validate_schema({"status": "open", "count": 5}, schema) == []
    errs = validate_schema({"status": "pending", "count": 99}, schema)
    assert len(errs) == 2


def test_refusal_detection():
    assert is_refusal("I'm sorry, but I don't have that information.")
    assert is_refusal("I cannot share that.")
    assert not is_refusal("Refunds are available within 30 days.")


def test_refusal_check_expected_true(tiny_corpus):
    ctx = _ctx("I'm sorry, I don't have that information.", tiny_corpus)
    assert run_check(CheckSpec("refusal", {"expected": True}), ctx).passed
    assert not run_check(CheckSpec("refusal", {"expected": False}), ctx).passed


def test_refusal_check_expected_false(tiny_corpus):
    ctx = _ctx("Refunds are available within 30 days.", tiny_corpus)
    assert run_check(CheckSpec("refusal", {"expected": False}), ctx).passed
