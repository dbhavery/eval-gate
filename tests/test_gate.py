"""Tests for the release-gate logic and exit codes."""

from __future__ import annotations

from eval_gate.config import load_suite
from eval_gate.gate import EXIT_GATE_FAILED, EXIT_OK, evaluate_gate
from eval_gate.runner import run_suite
from tests.conftest import FIXTURES


def test_passing_suite_gate_ok():
    suite = load_suite(FIXTURES / "suite.yaml")
    result = run_suite(suite)
    verdict = evaluate_gate(result, baseline=None)
    assert verdict.passed
    assert verdict.exit_code == EXIT_OK


def test_threshold_failure():
    suite = load_suite(FIXTURES / "suite_regressed.yaml")
    result = run_suite(suite)
    verdict = evaluate_gate(result, baseline=None)
    assert not verdict.passed
    assert verdict.exit_code == EXIT_GATE_FAILED
    assert any("threshold" in r for r in verdict.reasons)


def test_baseline_regression_detected():
    good = run_suite(load_suite(FIXTURES / "suite.yaml"))
    baseline = evaluate_and_dict(good)
    regressed = run_suite(load_suite(FIXTURES / "suite_regressed.yaml"))
    verdict = evaluate_gate(regressed, baseline=baseline)
    assert not verdict.passed
    assert set(verdict.regressions) == {"refund_window", "warranty_length", "privacy_data_sale"}
    assert any("regressed" in r for r in verdict.reasons)


def test_no_regression_when_compared_to_self():
    good = run_suite(load_suite(FIXTURES / "suite.yaml"))
    baseline = evaluate_and_dict(good)
    verdict = evaluate_gate(good, baseline=baseline)
    assert verdict.passed
    assert verdict.regressions == []


def evaluate_and_dict(result):
    """Helper: produce a baseline dict from a suite result."""
    return result.to_dict()
