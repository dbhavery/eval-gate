"""Command-line interface for eval-gate.

Subcommands:
    run        Run a suite, apply the gate, write reports, exit non-zero on fail.
    baseline   Run a suite and save its report as a baseline for future gating.
    list-checks  Print all registered check types.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval_gate import __version__
from eval_gate.checks import registered_checks
from eval_gate.config import ConfigError, load_suite
from eval_gate.gate import EXIT_ERROR, EXIT_OK, evaluate_gate, load_baseline
from eval_gate.providers import ProviderError
from eval_gate.report import write_html, write_json
from eval_gate.runner import SuiteResult, run_suite

_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{_RESET}" if use_color else text


def _print_summary(result: SuiteResult, verdict, use_color: bool) -> None:
    print(f"\n{_c('eval-gate', _BOLD, use_color)}  suite: {result.name}  provider: {result.provider}\n")
    for c in result.cases:
        mark = _c("PASS", _GREEN, use_color) if c.passed else _c("FAIL", _RED, use_color)
        print(f"  [{mark}] {c.id}: {c.question}")
        for ck in c.checks:
            if not ck.passed:
                print(f"         {_c('x', _RED, use_color)} {ck.check_type}: {ck.message}")
    rate = f"{result.pass_rate * 100:.0f}%"
    print(
        f"\n  cases {result.n_passed}/{result.n_cases}  "
        f"checks {result.passed_checks}/{result.total_checks}  "
        f"pass-rate {rate}  threshold {result.threshold * 100:.0f}%"
    )
    if verdict.passed:
        print(f"\n  {_c('GATE PASSED', _GREEN + _BOLD, use_color)}\n")
    else:
        print(f"\n  {_c('GATE FAILED', _RED + _BOLD, use_color)}")
        for r in verdict.reasons:
            print(f"    - {r}")
        print()


def _cmd_run(args: argparse.Namespace) -> int:
    use_color = not args.no_color and sys.stdout.isatty()
    try:
        suite = load_suite(args.suite)
        result = run_suite(suite, provider_mode=args.provider)
    except (ConfigError, ProviderError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    baseline = None
    if args.baseline:
        try:
            baseline = load_baseline(args.baseline)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR

    verdict = evaluate_gate(result, baseline)

    if not args.no_report:
        report_dir = Path(args.report_dir)
        json_path = write_json(result, report_dir / "report.json", verdict)
        html_path = write_html(result, verdict, report_dir / "report.html")
        print(f"{_c('wrote', _DIM, use_color)} {json_path}")
        print(f"{_c('wrote', _DIM, use_color)} {html_path}")

    _print_summary(result, verdict, use_color)
    return verdict.exit_code


def _cmd_baseline(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.suite)
        result = run_suite(suite, provider_mode=args.provider)
    except (ConfigError, ProviderError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    verdict = evaluate_gate(result, None)
    # A baseline is the record of what "good" looked like. Recording a failing run
    # freezes its failures as expected and every later run compares clean against
    # them, which silently disables the gate (2026-09-19 audit, finding D8).
    if not verdict.passed:
        print("error: refusing to record a baseline from a run that does not pass "
              "the gate:", file=sys.stderr)
        for r in verdict.reasons:
            print(f"  - {r}", file=sys.stderr)
        print("Fix the failures first, then re-record.", file=sys.stderr)
        return EXIT_ERROR
    out = write_json(result, args.out, verdict)
    print(f"wrote baseline: {out}  (pass-rate {result.pass_rate * 100:.0f}%)")
    return EXIT_OK


def _cmd_list_checks(args: argparse.Namespace) -> int:
    for name in registered_checks():
        print(name)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_gate",
        description="Evaluation release gate for LLM and RAG systems.",
    )
    p.add_argument("--version", action="version", version=f"eval-gate {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite and apply the release gate")
    run.add_argument("--suite", required=True, help="path to a suite YAML file")
    run.add_argument(
        "--provider",
        default="deterministic",
        choices=["deterministic", "auto", "openai", "anthropic"],
        help="answer source (default: deterministic, offline)",
    )
    run.add_argument("--baseline", help="baseline report JSON to gate regressions against")
    run.add_argument("--report-dir", default="reports", help="directory for JSON/HTML reports")
    run.add_argument("--no-report", action="store_true", help="skip writing report files")
    run.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    run.set_defaults(func=_cmd_run)

    base = sub.add_parser("baseline", help="run a suite and save its report as a baseline")
    base.add_argument("--suite", required=True, help="path to a suite YAML file")
    base.add_argument("--provider", default="deterministic",
                      choices=["deterministic", "auto", "openai", "anthropic"])
    base.add_argument("--out", required=True, help="output baseline JSON path")
    base.set_defaults(func=_cmd_baseline)

    lc = sub.add_parser("list-checks", help="list registered check types")
    lc.set_defaults(func=_cmd_list_checks)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
