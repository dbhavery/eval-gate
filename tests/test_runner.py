"""Tests for the suite runner over the real fixtures."""

from __future__ import annotations

from eval_gate.config import load_suite
from eval_gate.runner import run_suite
from tests.conftest import FIXTURES


def test_passing_suite_all_green():
    suite = load_suite(FIXTURES / "suite.yaml")
    result = run_suite(suite, provider_mode="deterministic")
    assert result.n_cases == 13
    assert result.n_passed == 13
    assert result.pass_rate == 1.0
    assert result.failing_case_ids() == []
    assert result.provider == "deterministic"


def test_run_is_reproducible():
    suite = load_suite(FIXTURES / "suite.yaml")
    a = run_suite(suite).to_dict()
    b = run_suite(suite).to_dict()
    # answers, retrieval, and per-check results must match byte-for-byte
    a.pop("cases"), b.pop("cases")  # timestamps not in to_dict, safe to compare
    assert run_suite(suite).to_dict()["cases"] == run_suite(suite).to_dict()["cases"]


def test_regressed_suite_has_failures():
    suite = load_suite(FIXTURES / "suite_regressed.yaml")
    result = run_suite(suite, provider_mode="deterministic")
    failing = set(result.failing_case_ids())
    assert {"refund_window", "warranty_length", "privacy_data_sale"} == failing
    assert result.pass_rate < 1.0


def test_retrieval_populated_for_grounded_case():
    suite = load_suite(FIXTURES / "suite.yaml")
    result = run_suite(suite)
    by_id = {c.id: c for c in result.cases}
    assert "refund_policy" in by_id["refund_window"].retrieved_ids
