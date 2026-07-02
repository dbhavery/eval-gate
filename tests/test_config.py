"""Tests for suite YAML loading and validation."""

from __future__ import annotations

import pytest

from eval_gate.config import ConfigError, load_suite
from tests.conftest import FIXTURES


def test_load_real_suite():
    suite = load_suite(FIXTURES / "suite.yaml")
    assert suite.name == "support-assistant-eval"
    assert suite.threshold == 0.85
    assert len(suite.cases) == 13
    assert suite.cases[0].id == "refund_window"
    assert suite.cases[0].checks[0].type == "recall_at_k"


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_suite(FIXTURES / "does_not_exist.yaml")


def test_bad_threshold_raises(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("name: x\nthreshold: 5\ncases:\n  - id: a\n    question: q\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_suite(p)


def test_duplicate_case_id_raises(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "name: x\ncases:\n  - id: a\n    question: q1\n  - id: a\n    question: q2\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_suite(p)


def test_empty_cases_raises(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("name: x\ncases: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_suite(p)
