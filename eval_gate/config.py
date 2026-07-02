"""Suite configuration: typed dataclasses plus a YAML loader.

A *suite* is a versionable, diffable description of what "good" looks like for a
set of questions against a document corpus. It is plain data — no logic — so it
can be reviewed in a pull request by anyone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a suite file is malformed or references missing data."""


@dataclass(frozen=True)
class CheckSpec:
    """A single check to run against a case's model output / retrieval.

    Attributes:
        type: Check identifier, e.g. ``must_contain``, ``recall_at_k``.
        params: Free-form parameters consumed by the check implementation.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    """One evaluation case: a question plus its graded expectations.

    Attributes:
        id: Stable, unique case identifier (also the response fixture key).
        question: The user question posed to the RAG system.
        relevant_ids: Ground-truth relevant corpus document ids (for retrieval
            precision/recall/grounding). May be empty for pure-generation cases.
        top_k: Number of documents the retriever should return for this case.
        checks: Ordered list of checks that must all pass for the case to pass.
    """

    id: str
    question: str
    relevant_ids: list[str] = field(default_factory=list)
    top_k: int = 3
    checks: list[CheckSpec] = field(default_factory=list)


@dataclass(frozen=True)
class SuiteSpec:
    """A full evaluation suite.

    Attributes:
        name: Human-readable suite name.
        corpus_dir: Directory (relative to the suite file) holding corpus docs.
        responses_dir: Directory holding recorded deterministic model outputs.
        threshold: Minimum overall pass-rate in [0, 1] required to pass the gate.
        cases: The evaluation cases.
        path: Absolute path the suite was loaded from (for resolving fixtures).
    """

    name: str
    corpus_dir: str
    responses_dir: str
    threshold: float
    cases: list[CaseSpec]
    path: Path

    def resolve(self, sub: str) -> Path:
        """Resolve a suite-relative path to an absolute one."""
        return (self.path.parent / sub).resolve()


def _require(mapping: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{ctx}: missing required key '{key}'")
    return mapping[key]


def load_suite(path: str | Path) -> SuiteSpec:
    """Load and validate a suite from a YAML file.

    Args:
        path: Path to the suite YAML file.

    Returns:
        A validated :class:`SuiteSpec`.

    Raises:
        ConfigError: If the file is missing, unparseable, or malformed.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise ConfigError(f"suite file not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - passthrough of parser msg
        raise ConfigError(f"could not parse YAML in {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"suite root must be a mapping, got {type(raw).__name__}")

    name = str(_require(raw, "name", "suite"))
    corpus_dir = str(raw.get("corpus_dir", "corpus"))
    responses_dir = str(raw.get("responses_dir", "responses"))

    threshold = raw.get("threshold", 1.0)
    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        raise ConfigError(f"suite 'threshold' must be a number in [0, 1], got {threshold!r}")

    raw_cases = _require(raw, "cases", "suite")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigError("suite 'cases' must be a non-empty list")

    cases: list[CaseSpec] = []
    seen_ids: set[str] = set()
    for i, rc in enumerate(raw_cases):
        ctx = f"case[{i}]"
        if not isinstance(rc, dict):
            raise ConfigError(f"{ctx} must be a mapping")
        cid = str(_require(rc, "id", ctx))
        if cid in seen_ids:
            raise ConfigError(f"duplicate case id '{cid}'")
        seen_ids.add(cid)

        question = str(_require(rc, "question", ctx))
        relevant_ids = [str(x) for x in rc.get("relevant_ids", [])]
        top_k = int(rc.get("top_k", 3))
        if top_k < 1:
            raise ConfigError(f"{ctx}: top_k must be >= 1")

        raw_checks = rc.get("checks", [])
        if not isinstance(raw_checks, list):
            raise ConfigError(f"{ctx}: 'checks' must be a list")
        checks: list[CheckSpec] = []
        for j, rk in enumerate(raw_checks):
            if not isinstance(rk, dict):
                raise ConfigError(f"{ctx}.checks[{j}] must be a mapping")
            ctype = str(_require(rk, "type", f"{ctx}.checks[{j}]"))
            params = {k: v for k, v in rk.items() if k != "type"}
            checks.append(CheckSpec(type=ctype, params=params))

        cases.append(
            CaseSpec(
                id=cid,
                question=question,
                relevant_ids=relevant_ids,
                top_k=top_k,
                checks=checks,
            )
        )

    return SuiteSpec(
        name=name,
        corpus_dir=corpus_dir,
        responses_dir=responses_dir,
        threshold=float(threshold),
        cases=cases,
        path=p,
    )
