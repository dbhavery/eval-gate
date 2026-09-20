"""Prove this test suite can fail.

Every other test here checks that the gate behaves correctly on a fixture. None
of them answers the question a reviewer should actually ask: *if the checking
code silently broke, would anything notice?*

A suite that stays green after you break the code is a smoke alarm with no
battery, and the 2026-09-19 audit found exactly that shape of defect three times
in this repository. So this module sabotages the source on purpose, one edit at
a time, and asserts the rest of the suite goes red.

Each mutation below is a real defect that existed here, or a plausible next one.
For each, the harness copies the package to a temporary directory, applies the
edit, runs the suite there, and requires a non-zero exit. A mutation that
survives means some behaviour is unverified, and the test names the line.

The control at the bottom runs the same harness with no edit applied and
requires a clean pass. Without it, a harness that always reported failure would
look like a suite that catches everything.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Directories that are large, generated, or irrelevant to running the suite.
_SKIP = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", "*.egg-info", "reports",
    "screenshots", ".pytest_cache", ".mypy_cache", ".ruff_cache",
)


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect, and the behaviour that should catch it."""

    name: str
    path: str
    old: str
    new: str
    why: str


MUTATIONS = [
    Mutation(
        name="a case with no checks passes again",
        path="eval_gate/runner.py",
        old="return bool(self.checks) and all(c.passed for c in self.checks)",
        new="return all(c.passed for c in self.checks)",
        why=(
            "audit finding D1. all([]) is True, so an unchecked case used to "
            "count as a pass and inflate the suite pass-rate."
        ),
    ),
    Mutation(
        name="a failing case no longer fails the gate",
        path="eval_gate/gate.py",
        old='    if failing:\n        reasons.append(f"{len(failing)} case(s) failed: {failing}")',
        new='    if False:\n        reasons.append(f"{len(failing)} case(s) failed: {failing}")',
        why=(
            "rule 1 is the whole product. With it gone the gate leans on the "
            "pass-rate threshold alone, which is what let a wrong answer "
            "through at threshold 0.85."
        ),
    ),
    Mutation(
        name="the pass-rate threshold is ignored",
        path="eval_gate/gate.py",
        old="    if result.pass_rate < result.threshold - _EPS:",
        new="    if False:",
        why="the second bar stops being a bar.",
    ),
    Mutation(
        name="a drop against the baseline is ignored",
        path="eval_gate/gate.py",
        old="        if result.pass_rate < base_rate - _RATE_TOLERANCE:",
        new="        if False:",
        why=(
            "the README's headline promise is that the gate exits non-zero on "
            "regression. This is the comparison that keeps that promise."
        ),
    ),
    Mutation(
        name="negation stops changing the meaning of an answer",
        path="eval_gate/checks/quality.py",
        old='_NEGATION_CUES = frozenset(\n    """\n',
        new='_NEGATION_CUES = frozenset()\n_UNUSED_CUES = frozenset(\n    """\n',
        why=(
            "audit finding D3. With negation treated as filler, 'we do not "
            "sell personal data' and 'we sell personal data' both scored "
            "1.000 faithfulness against the same context."
        ),
    ),
    Mutation(
        name="must_contain always passes",
        path="eval_gate/checks/assertions.py",
        old="    ok = needle in hay\n",
        new="    ok = True\n",
        why="the simplest assertion in the product, stubbed out.",
    ),
]


def _run_suite_in(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the suite inside *root*, with this module excluded.

    Excluding this file is load bearing rather than tidying: without it the
    inner run would re-enter the harness and recurse.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
         "--ignore=tests/test_mutation.py", "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="module")
def pristine(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One clean copy of the repository, reused by every mutation."""
    dest = tmp_path_factory.mktemp("pristine") / "eval-gate"
    shutil.copytree(REPO, dest, ignore=_SKIP)
    return dest


def _copy_and_mutate(pristine: Path, tmp_path: Path, mut: Mutation) -> Path:
    work = tmp_path / "repo"
    shutil.copytree(pristine, work, ignore=_SKIP)
    target = work / mut.path
    src = target.read_text(encoding="utf-8")

    # Refuse to guess. An anchor that is missing or ambiguous means the source
    # moved, and a mutation applied to the wrong line would prove nothing.
    occurrences = src.count(mut.old)
    assert occurrences == 1, (
        f"mutation {mut.name!r}: anchor appears {occurrences} times in "
        f"{mut.path}, expected exactly 1. The source moved; update the anchor."
    )
    target.write_text(src.replace(mut.old, mut.new, 1), encoding="utf-8")
    return work


def test_control_the_unmutated_suite_passes(pristine: Path, tmp_path: Path) -> None:
    """The harness reports a clean pass when nothing is broken.

    This is the working control. Every assertion below is of the form "the
    suite failed", which a permanently broken harness would also satisfy.
    """
    work = tmp_path / "repo"
    shutil.copytree(pristine, work, ignore=_SKIP)
    proc = _run_suite_in(work)
    assert proc.returncode == 0, (
        "the unmutated suite did not pass, so every mutation result below is "
        f"meaningless.\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}"
    )


@pytest.mark.parametrize("mut", MUTATIONS, ids=lambda m: m.name)
def test_mutation_is_caught(pristine: Path, tmp_path: Path, mut: Mutation) -> None:
    """Breaking this line must turn the suite red."""
    work = _copy_and_mutate(pristine, tmp_path, mut)
    proc = _run_suite_in(work)
    assert proc.returncode != 0, (
        f"MUTATION SURVIVED: {mut.name}\n"
        f"  {mut.path}: {mut.old.strip()!r} -> {mut.new.strip()!r}\n"
        f"  why it matters: {mut.why}\n"
        f"  The suite passed with this defect in place, so nothing here "
        f"verifies that behaviour.\n{proc.stdout[-3000:]}"
    )
