"""Regression tests for the 2026-09-19 audit findings.

Each test in this file was written *before* the corresponding fix, run against
the unfixed code, and observed to fail. Every one of them describes a concrete
way a bad answer could pass the gate (or, for the last two, a way a good run
could be mis-reported). The test name states the rule that must hold.

Run just this file with::

    pytest tests/test_audit_2026_09_19.py -q
"""

from __future__ import annotations

import json

import pytest

from eval_gate.checks import CheckContext, run_check
from eval_gate.checks.assertions import validate_schema
from eval_gate.checks.quality import faithfulness_score
from eval_gate.cli import main
from eval_gate.config import CheckSpec, ConfigError, load_suite
from eval_gate.corpus import Corpus, Document
from eval_gate.gate import evaluate_gate, load_baseline
from eval_gate.runner import CaseResult, CheckResult, SuiteResult, run_suite
from tests.conftest import FIXTURES

PRIVACY_DOC = (FIXTURES / "corpus" / "data_privacy.md").read_text(encoding="utf-8")


def _suite_file(tmp_path, body: str):
    """Write a suite YAML that points at the real corpus and responses."""
    p = tmp_path / "suite.yaml"
    p.write_text(
        f"name: audit-suite\n"
        f"corpus_dir: {(FIXTURES / 'corpus').as_posix()}\n"
        f"responses_dir: {(FIXTURES / 'responses').as_posix()}\n"
        f"{body}",
        encoding="utf-8",
    )
    return p


def _case(cid: str, check_flags: list[bool]) -> CaseResult:
    return CaseResult(
        id=cid,
        question="q",
        answer="a",
        retrieved_ids=[],
        relevant_ids=[],
        checks=[CheckResult("must_contain", ok, "m") for ok in check_flags],
    )


# --------------------------------------------------------------------------
# D1 - a case with no checks must not score as a pass
# --------------------------------------------------------------------------


def test_d1_case_with_no_checks_is_not_a_pass():
    """runner.py: `all([]) is True` made an unchecked case a silent pass."""
    assert _case("empty", []).passed is False


def test_d1_suite_with_an_uncheckable_case_is_refused_at_load_time(tmp_path):
    p = _suite_file(tmp_path, "threshold: 1.0\ncases:\n  - id: refund_window\n    question: q\n")
    with pytest.raises(ConfigError, match="at least one check"):
        load_suite(p)


# --------------------------------------------------------------------------
# D2 - one wholly failing case must not pass the gate
# --------------------------------------------------------------------------


def test_d2_a_single_failing_case_fails_the_gate():
    """12/13 = 0.923 cleared the 0.85 threshold while one answer was entirely wrong."""
    cases = [_case(f"ok{i}", [True, True]) for i in range(12)]
    cases.append(_case("broken", [True, False, False, False]))
    result = SuiteResult(name="s", provider="deterministic", threshold=0.85, cases=cases)
    assert result.pass_rate > 0.85  # the old gate's only rule was satisfied
    verdict = evaluate_gate(result, baseline=None)
    assert not verdict.passed
    assert verdict.exit_code == 1
    assert any("broken" in r for r in verdict.reasons)


# --------------------------------------------------------------------------
# D3 - negation must not be stripped before a faithfulness comparison
# --------------------------------------------------------------------------


def test_d3_dropping_a_negation_lowers_faithfulness():
    """"We do not sell personal data" and "We sell personal data" both scored 1.000."""
    grounded = faithfulness_score("We do not sell personal data to third parties.", PRIVACY_DOC)
    inverted = faithfulness_score("We sell personal data to third parties.", PRIVACY_DOC)
    assert grounded == pytest.approx(1.0)
    assert inverted < 0.8


def test_d3_inventing_a_negation_lowers_faithfulness():
    """The opposite direction: a negation the context does not support."""
    doc = "Deletion requests are completed within 30 days."
    assert faithfulness_score("Deletion requests are not completed within 30 days.", doc) < 0.8


# --------------------------------------------------------------------------
# D4 - a contentless answer is not a faithful answer
# --------------------------------------------------------------------------


def test_d4_contentless_answer_is_not_faithful():
    assert faithfulness_score("", PRIVACY_DOC) == 0.0
    assert faithfulness_score("the and of", PRIVACY_DOC) == 0.0
    corpus = Corpus([Document(id="alpha", text="Refunds are available within 30 days.")])
    ctx = CheckContext(answer="   ", retrieved_ids=["alpha"], relevant_ids=[], corpus=corpus)
    assert not run_check(CheckSpec("faithfulness", {"min": 0.8}), ctx).passed


# --------------------------------------------------------------------------
# D5 - an unknown check type is a config error (exit 2), not a gate failure
# --------------------------------------------------------------------------


def test_d5_unknown_check_type_is_refused_at_load_time(tmp_path):
    p = _suite_file(
        tmp_path,
        'threshold: 1.0\ncases:\n  - id: refund_window\n    question: q\n'
        '    checks:\n      - type: must_contian\n        value: "30 days"\n',
    )
    with pytest.raises(ConfigError, match="unknown check type"):
        load_suite(p)


def test_d5_unknown_check_type_exits_two_not_one(tmp_path):
    p = _suite_file(
        tmp_path,
        'threshold: 1.0\ncases:\n  - id: refund_window\n    question: q\n'
        '    checks:\n      - type: must_contian\n        value: "30 days"\n',
    )
    assert main(["run", "--suite", str(p), "--no-report", "--no-color"]) == 2


# --------------------------------------------------------------------------
# D6 - a check that cannot fail must be refused, not run
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        CheckSpec("length", {}),                                  # no bounds at all
        CheckSpec("length", {"max_word": 60}),                    # misspelled bound
        CheckSpec("must_contain", {"value": ""}),                 # empty needle
        CheckSpec("must_not_contain", {"value": ""}),             # empty needle
        CheckSpec("regex", {"pattern": ""}),                      # matches everything
        CheckSpec("regex", {"pattern": "("}),                     # not a valid regex
        CheckSpec("forbidden_phrases", {"phrases": []}),          # nothing forbidden
        CheckSpec("forbidden_phrases", {"phrases": [""]}),        # empty phrase
        CheckSpec("faithfulness", {"minimum": 0.9}),              # misspelled 'min'
        CheckSpec("faithfulness", {"min": 1.7}),                  # out of range
        CheckSpec("recall_at_k", {"k": 0, "min": 1.0}),           # k must be >= 1
        CheckSpec("format", {"kind": "yaml"}),                    # unknown kind
        CheckSpec("must_contain", {"value": "x", "ci": True}),    # unknown parameter
    ],
)
def test_d6_unusable_check_specs_are_rejected(spec):
    from eval_gate.checks import validate_check_spec

    assert validate_check_spec(spec), f"{spec} should have been rejected"


def test_d6_a_boundless_length_check_cannot_fail_and_is_refused(tmp_path):
    p = _suite_file(
        tmp_path,
        "threshold: 1.0\ncases:\n  - id: refund_window\n    question: q\n"
        "    checks:\n      - type: length\n",
    )
    with pytest.raises(ConfigError):
        load_suite(p)


def test_d6_every_registered_check_declares_its_parameters():
    """A new check must declare its params, or this guard fails instead of drifting."""
    from eval_gate.checks import _REGISTRY, declared_params

    for name in _REGISTRY:
        assert declared_params(name) is not None, f"check '{name}' declares no parameters"


# --------------------------------------------------------------------------
# D7 - json_schema must not silently ignore keywords it cannot enforce
# --------------------------------------------------------------------------


def test_d7_unsupported_schema_keywords_are_not_silently_ignored():
    from eval_gate.checks.assertions import SchemaDefinitionError

    with pytest.raises(SchemaDefinitionError):
        validate_schema({"a": 1, "junk": 2}, {"type": "object", "additionalProperties": False})
    with pytest.raises(SchemaDefinitionError):
        validate_schema("zzz", {"type": "string", "pattern": "^a+$"})
    with pytest.raises(SchemaDefinitionError):
        validate_schema([1, 1, 1], {"type": "array", "minItems": 9, "uniqueItems": True})


def test_d7_unsupported_schema_keyword_is_refused_at_load_time(tmp_path):
    p = _suite_file(
        tmp_path,
        "threshold: 1.0\ncases:\n  - id: refund_window_json\n    question: q\n"
        "    checks:\n      - type: json_schema\n        schema:\n"
        "          type: object\n          additionalProperties: false\n",
    )
    with pytest.raises(ConfigError):
        load_suite(p)


# --------------------------------------------------------------------------
# D8 - a baseline must not be recorded from a failing run
# --------------------------------------------------------------------------


def test_d8_baseline_from_a_failing_run_is_refused(tmp_path):
    out = tmp_path / "baseline.json"
    code = main(["baseline", "--suite", str(FIXTURES / "suite_regressed.yaml"), "--out", str(out)])
    assert code == 2
    assert not out.exists(), "a failing run must not be frozen as the expected state"


# --------------------------------------------------------------------------
# D9 - a malformed baseline must not silently disable regression detection
# --------------------------------------------------------------------------


def test_d9_empty_baseline_is_rejected(tmp_path):
    from eval_gate.gate import BaselineError

    p = tmp_path / "b.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(BaselineError):
        load_baseline(p)


def test_d9_malformed_baseline_exits_two(tmp_path):
    p = tmp_path / "b.json"
    p.write_text('{"cases": []}', encoding="utf-8")
    code = main([
        "run", "--suite", str(FIXTURES / "suite.yaml"),
        "--baseline", str(p), "--no-report", "--no-color",
    ])
    assert code == 2


# --------------------------------------------------------------------------
# D10 - deleting a check is a regression, not a pass
# --------------------------------------------------------------------------


def test_d10_removing_a_check_is_reported_as_a_regression():
    baseline = SuiteResult(
        name="s", provider="deterministic", threshold=1.0,
        cases=[
            CaseResult("refund_window", "q", "a", [], [],
                       [CheckResult("must_contain", True, "m"),
                        CheckResult("faithfulness", True, "m"),
                        CheckResult("citations_grounded", True, "m")]),
        ],
    ).to_dict()
    weakened = SuiteResult(
        name="s", provider="deterministic", threshold=1.0,
        cases=[CaseResult("refund_window", "q", "a", [], [],
                          [CheckResult("must_contain", True, "m")])],
    )
    verdict = evaluate_gate(weakened, baseline=baseline)
    assert not verdict.passed
    assert any("faithfulness" in r or "check" in r for r in verdict.reasons)


# --------------------------------------------------------------------------
# D11 - the bar cannot be lowered between the baseline and the run
# --------------------------------------------------------------------------


def test_d11_lowering_the_threshold_below_the_baseline_fails_the_gate():
    cases = [_case("ok", [True])]
    baseline = SuiteResult("s", "deterministic", 1.0, cases).to_dict()
    lowered = SuiteResult("s", "deterministic", 0.5, cases)
    verdict = evaluate_gate(lowered, baseline=baseline)
    assert not verdict.passed
    assert any("threshold" in r for r in verdict.reasons)


def test_d11_a_zero_threshold_is_refused(tmp_path):
    p = _suite_file(
        tmp_path,
        'threshold: 0.0\ncases:\n  - id: refund_window\n    question: q\n'
        '    checks:\n      - type: must_contain\n        value: "30 days"\n',
    )
    with pytest.raises(ConfigError, match="threshold"):
        load_suite(p)


# --------------------------------------------------------------------------
# D12 - an identical re-run is not a regression (rounding in the baseline)
# --------------------------------------------------------------------------


def test_d12_identical_rerun_is_not_reported_as_a_drop():
    """to_dict() rounds pass_rate to 6dp; 10/13 round-trips *above* its true value."""
    cases = [_case(f"ok{i}", [True]) for i in range(10)] + [
        _case(f"bad{i}", [False]) for i in range(3)
    ]
    result = SuiteResult("s", "deterministic", 0.5, cases)
    baseline = json.loads(json.dumps(result.to_dict()))
    verdict = evaluate_gate(result, baseline=baseline)
    assert not any("dropped below baseline" in r for r in verdict.reasons), verdict.reasons


# --------------------------------------------------------------------------
# D13 - a pass-rate drop against the baseline is reported as its own reason
#
# Found 2026-09-20 by tests/test_mutation.py, not by a human reading the code.
# Replacing the comparison in gate.py with `if False:` left all 89 tests green,
# which meant nothing here verified the behaviour the README leads with.
#
# The rule can never be the *sole* cause of a failure: pass_rate is
# n_passed / n_cases, so any rate below 1.0 means a case failed, and rule 1
# already fires for that. What it adds is the sentence a human reads in the
# report, naming the size of the drop. That sentence is worth asserting.
# --------------------------------------------------------------------------


def test_d13_a_pass_rate_drop_against_the_baseline_is_named_in_the_reasons():
    """A newly added failing case drops the rate without regressing any case."""
    good = [_case(f"ok{i}", [True]) for i in range(4)]
    baseline = SuiteResult("s", "deterministic", 1.0, good).to_dict()

    # Same four cases, still passing, plus one new failure. Nothing regressed.
    current = SuiteResult("s", "deterministic", 1.0, good + [_case("bad", [False])])
    verdict = evaluate_gate(current, baseline=baseline)

    assert not verdict.passed
    assert verdict.regressions == [], (
        "no baseline case changed state, so this must not be called a regression"
    )
    assert any("dropped below baseline" in r for r in verdict.reasons), verdict.reasons


# --------------------------------------------------------------------------
# Controls: the fixes must not break the real suites.
# --------------------------------------------------------------------------


def test_control_passing_suite_still_passes_clean():
    result = run_suite(load_suite(FIXTURES / "suite.yaml"))
    assert result.n_passed == result.n_cases == 13
    assert evaluate_gate(result, baseline=load_baseline(FIXTURES / "baseline.json")).passed


def test_control_regressed_suite_still_fails():
    result = run_suite(load_suite(FIXTURES / "suite_regressed.yaml"))
    verdict = evaluate_gate(result, baseline=load_baseline(FIXTURES / "baseline.json"))
    assert not verdict.passed
    assert verdict.exit_code == 1
