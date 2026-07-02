"""Tests for JSON and HTML report writers."""

from __future__ import annotations

import json

from eval_gate.config import load_suite
from eval_gate.gate import evaluate_gate
from eval_gate.report import render_html, write_html, write_json
from eval_gate.runner import run_suite
from tests.conftest import FIXTURES


def _result_and_verdict(suite_name="suite.yaml"):
    suite = load_suite(FIXTURES / suite_name)
    result = run_suite(suite)
    return result, evaluate_gate(result, None)


def test_write_json_has_gate_and_cases(tmp_path):
    result, verdict = _result_and_verdict()
    out = write_json(result, tmp_path / "r.json", verdict)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["gate"]["passed"] is True
    assert data["gate"]["exit_code"] == 0
    assert len(data["cases"]) == 13
    assert "generated_at" in data


def test_html_contains_verdict_and_is_self_contained(tmp_path):
    result, verdict = _result_and_verdict()
    html = render_html(result, verdict)
    assert "GATE PASSED" in html
    assert "<style>" in html
    # no external assets / CDNs
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn" not in html.lower()


def test_html_failing_verdict_lists_reasons(tmp_path):
    result, verdict = _result_and_verdict("suite_regressed.yaml")
    out = write_html(result, verdict, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "GATE FAILED" in html
    assert "threshold" in html
