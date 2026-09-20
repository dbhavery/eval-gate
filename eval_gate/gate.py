"""Release-gate logic: turn a suite result into a pass/fail verdict + exit code.

The gate fails (non-zero exit) when any of these hold:

1. **any case failed.** A case is a requirement, so one failing case fails the
   gate. The pass-rate threshold is a second, weaker bar and cannot excuse a
   failing case: at 13 cases and ``threshold: 0.85`` a wholly wrong answer used
   to clear the gate at 12/13 = 0.923 (2026-09-19 audit, finding D2).
2. the overall pass-rate is below the suite threshold,
3. a baseline is supplied and the run *regressed* against it: a case that passed
   in the baseline now fails, or the pass-rate dropped,
4. a baseline is supplied and the *suite* was weakened rather than the system
   fixed: a case lost checks, or the declared threshold was lowered (findings
   D10 and D11).

This is what makes the tool a CI gate rather than just a report generator.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_gate.runner import SuiteResult

# Exit codes (documented so CI can distinguish gate-fail from tool-error).
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2

_EPS = 1e-9

# SuiteResult.to_dict() rounds pass_rate to 6 decimal places, so a re-recorded
# identical run can round *up* by up to 5e-7. Comparing against a stored rate
# with an epsilon smaller than that reported an identical re-run as a drop
# (2026-09-19 audit, finding D12).
_RATE_TOLERANCE = 1e-6


class BaselineError(ValueError):
    """Raised when a baseline file is not a usable baseline.

    A baseline the gate cannot read is not a reason to skip the comparison. An
    empty ``{}`` used to load fine and silently turn regression detection off
    (2026-09-19 audit, finding D9), so this is raised instead and the CLI maps it
    to ``EXIT_ERROR``.
    """


@dataclass
class GateVerdict:
    """Outcome of applying the gate to a suite result.

    Attributes:
        passed: Whether the gate passed.
        reasons: Human-readable reasons the gate failed (empty if passed).
        regressions: Case ids that passed in the baseline but now fail.
        exit_code: Process exit code to return.
    """

    passed: bool
    reasons: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_OK if self.passed else EXIT_GATE_FAILED


def validate_baseline(baseline: Any) -> None:
    """Raise :class:`BaselineError` unless *baseline* can actually gate a run.

    Checked: it is a mapping, it holds a non-empty ``cases`` list, every case has
    an ``id``, a ``passed`` flag and the ``check_types`` the gate needs to spot a
    weakened suite, and it records a numeric ``pass_rate`` and ``threshold``.
    """
    if not isinstance(baseline, dict):
        raise BaselineError(
            f"baseline must be a JSON object, got {type(baseline).__name__}"
        )
    cases = baseline.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BaselineError(
            "baseline has no cases; it cannot gate anything. Re-record it with "
            "`eval_gate baseline --suite <suite> --out <path>`"
        )
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BaselineError(f"baseline case[{i}] must be a JSON object")
        for key in ("id", "passed", "check_types"):
            if key not in case:
                raise BaselineError(
                    f"baseline case[{i}] is missing '{key}'. This baseline predates the "
                    "2026-09-19 gate rules; re-record it with `eval_gate baseline`"
                )
    for key in ("pass_rate", "threshold"):
        value = baseline.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BaselineError(f"baseline '{key}' must be a number, got {value!r}")


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load and validate a baseline JSON report produced by a prior passing run.

    Raises:
        FileNotFoundError: If the file does not exist.
        BaselineError: If the file is not usable as a baseline.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"baseline file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"baseline {p} is not valid JSON: {exc}") from exc
    validate_baseline(data)
    return data


def _baseline_case_pass_map(baseline: dict[str, Any]) -> dict[str, bool]:
    return {c["id"]: bool(c["passed"]) for c in baseline.get("cases", [])}


def evaluate_gate(result: SuiteResult, baseline: dict[str, Any] | None = None) -> GateVerdict:
    """Apply the gate to *result*, optionally comparing to a *baseline* report.

    Args:
        result: The current suite result.
        baseline: A previously-recorded report dict (from ``SuiteResult.to_dict``).

    Returns:
        A :class:`GateVerdict`.
    """
    reasons: list[str] = []

    # Rule 1: a failing case fails the gate. The threshold below is a second bar,
    # not an allowance that lets a wrong answer through.
    failing = result.failing_case_ids()
    if failing:
        reasons.append(f"{len(failing)} case(s) failed: {failing}")

    if result.pass_rate < result.threshold - _EPS:
        reasons.append(
            f"pass-rate {result.pass_rate:.3f} is below threshold {result.threshold:.3f}"
        )

    regressions: list[str] = []
    if baseline is not None:
        validate_baseline(baseline)
        base_pass = _baseline_case_pass_map(baseline)
        current_pass = {c.id: c.passed for c in result.cases}
        for cid, was_passing in base_pass.items():
            if was_passing and not current_pass.get(cid, False):
                regressions.append(cid)
        if regressions:
            reasons.append(f"{len(regressions)} case(s) regressed vs baseline: {regressions}")

        base_rate = float(baseline["pass_rate"])
        if result.pass_rate < base_rate - _RATE_TOLERANCE:
            reasons.append(
                f"pass-rate {result.pass_rate:.3f} dropped below baseline {base_rate:.3f}"
            )

        reasons.extend(_suite_weakening_reasons(result, baseline))

    return GateVerdict(passed=not reasons, reasons=reasons, regressions=regressions)


def _suite_weakening_reasons(result: SuiteResult, baseline: dict[str, Any]) -> list[str]:
    """Reasons the *suite* got easier, rather than the system getting better.

    Two ways to turn a red gate green without touching the answer: delete the
    check that was failing, or lower the declared threshold. Both are compared
    against the baseline here (2026-09-19 audit, findings D10 and D11).
    """
    reasons: list[str] = []

    current_types = {c.id: list(c.check_types) for c in result.cases}
    for case in baseline["cases"]:
        cid = case["id"]
        if cid not in current_types:
            continue  # a deleted case is already reported as a regression
        before = Counter(str(t) for t in case["check_types"])
        after = Counter(current_types[cid])
        removed = before - after
        if removed:
            dropped = sorted(removed.elements())
            reasons.append(f"case '{cid}' lost check(s) {dropped} that the baseline ran")

    base_threshold = float(baseline["threshold"])
    if result.threshold < base_threshold - _EPS:
        reasons.append(
            f"suite threshold {result.threshold:.3f} is below the baseline's "
            f"{base_threshold:.3f}; the bar cannot be lowered to pass"
        )
    return reasons
