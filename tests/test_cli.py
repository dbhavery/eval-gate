"""End-to-end CLI tests: these are the proof that the gate exit codes behave."""

from __future__ import annotations

from eval_gate.cli import main
from tests.conftest import FIXTURES


def test_cli_passing_run_exits_zero(tmp_path):
    code = main([
        "run",
        "--suite", str(FIXTURES / "suite.yaml"),
        "--report-dir", str(tmp_path),
        "--no-color",
    ])
    assert code == 0
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.html").is_file()


def test_cli_threshold_failure_exits_nonzero(tmp_path):
    code = main([
        "run",
        "--suite", str(FIXTURES / "suite_regressed.yaml"),
        "--report-dir", str(tmp_path),
        "--no-color",
    ])
    assert code == 1


def test_cli_baseline_regression_exits_nonzero(tmp_path):
    baseline = tmp_path / "baseline.json"
    assert main([
        "baseline", "--suite", str(FIXTURES / "suite.yaml"), "--out", str(baseline),
    ]) == 0
    code = main([
        "run",
        "--suite", str(FIXTURES / "suite_regressed.yaml"),
        "--baseline", str(baseline),
        "--report-dir", str(tmp_path),
        "--no-color",
    ])
    assert code == 1


def test_cli_unknown_suite_exits_error_code_2(tmp_path):
    code = main([
        "run", "--suite", str(tmp_path / "nope.yaml"), "--no-report", "--no-color",
    ])
    assert code == 2


def test_cli_list_checks_runs():
    assert main(["list-checks"]) == 0
