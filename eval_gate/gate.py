"""Release-gate logic: turn a suite result into a pass/fail verdict + exit code.

The gate fails (non-zero exit) when either:

1. the overall pass-rate is below the suite threshold, or
2. a baseline is supplied and the run *regressed* against it — the pass-rate
   dropped, or a case that passed in the baseline now fails.

This is what makes the tool a CI gate rather than just a report generator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_gate.runner import SuiteResult

# Exit codes (documented so CI can distinguish gate-fail from tool-error).
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2

_EPS = 1e-9


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


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load a baseline JSON report produced by a prior passing run."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"baseline file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


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

    if result.pass_rate < result.threshold - _EPS:
        reasons.append(
            f"pass-rate {result.pass_rate:.3f} is below threshold {result.threshold:.3f}"
        )

    regressions: list[str] = []
    if baseline is not None:
        base_pass = _baseline_case_pass_map(baseline)
        current_pass = {c.id: c.passed for c in result.cases}
        for cid, was_passing in base_pass.items():
            if was_passing and not current_pass.get(cid, False):
                regressions.append(cid)
        if regressions:
            reasons.append(f"{len(regressions)} case(s) regressed vs baseline: {regressions}")

        base_rate = float(baseline.get("pass_rate", 0.0))
        if result.pass_rate < base_rate - _EPS:
            reasons.append(
                f"pass-rate {result.pass_rate:.3f} dropped below baseline {base_rate:.3f}"
            )

    return GateVerdict(passed=not reasons, reasons=reasons, regressions=regressions)
