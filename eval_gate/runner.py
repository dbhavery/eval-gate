"""Suite runner: execute every case's checks and aggregate results.

Retrieval is always computed locally (deterministic TF-IDF). The answer text
comes from the configured provider (recorded fixtures by default, or a live LLM
when a provider key is set). A case passes only if *all* its checks pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from eval_gate.checks import CheckContext, CheckOutcome, run_check
from eval_gate.config import SuiteSpec
from eval_gate.corpus import Corpus, TfidfRetriever, load_corpus
from eval_gate.providers import Provider, build_provider


@dataclass
class CheckResult:
    """A check outcome flattened for reporting."""

    check_type: str
    passed: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_outcome(cls, o: CheckOutcome) -> "CheckResult":
        return cls(check_type=o.check_type, passed=o.passed, message=o.message, detail=o.detail)


@dataclass
class CaseResult:
    """Full result for one case."""

    id: str
    question: str
    answer: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) if self.checks else True

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)


@dataclass
class SuiteResult:
    """Aggregate result for a whole suite run."""

    name: str
    provider: str
    threshold: float
    cases: list[CaseResult]

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_cases if self.cases else 0.0

    @property
    def total_checks(self) -> int:
        return sum(len(c.checks) for c in self.cases)

    @property
    def passed_checks(self) -> int:
        return sum(c.n_passed for c in self.cases)

    def failing_case_ids(self) -> list[str]:
        return [c.id for c in self.cases if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON reporting / baselines."""
        return {
            "name": self.name,
            "provider": self.provider,
            "threshold": self.threshold,
            "pass_rate": round(self.pass_rate, 6),
            "n_cases": self.n_cases,
            "n_passed": self.n_passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "cases": [
                {
                    "id": c.id,
                    "question": c.question,
                    "answer": c.answer,
                    "passed": c.passed,
                    "retrieved_ids": c.retrieved_ids,
                    "relevant_ids": c.relevant_ids,
                    "checks": [asdict(ck) for ck in c.checks],
                }
                for c in self.cases
            ],
        }


def run_suite(
    suite: SuiteSpec,
    provider_mode: str = "deterministic",
    provider: Provider | None = None,
    corpus: Corpus | None = None,
) -> SuiteResult:
    """Run every case in *suite* and return an aggregated :class:`SuiteResult`.

    Args:
        suite: The loaded suite spec.
        provider_mode: Provider selection when *provider* is not supplied.
        provider: Optional explicit provider (mainly for tests).
        corpus: Optional preloaded corpus (mainly for tests).

    Returns:
        The aggregated suite result.
    """
    if corpus is None:
        corpus = load_corpus(suite.resolve(suite.corpus_dir))
    retriever = TfidfRetriever(corpus)
    if provider is None:
        provider = build_provider(provider_mode, suite.resolve(suite.responses_dir))

    case_results: list[CaseResult] = []
    for case in suite.cases:
        retrieved = retriever.retrieve(case.question, case.top_k)
        answer = provider.answer(case.id, case.question, retrieved, corpus)
        ctx = CheckContext(
            answer=answer,
            retrieved_ids=retrieved,
            relevant_ids=case.relevant_ids,
            corpus=corpus,
        )
        outcomes = [run_check(spec, ctx) for spec in case.checks]
        case_results.append(
            CaseResult(
                id=case.id,
                question=case.question,
                answer=answer,
                retrieved_ids=retrieved,
                relevant_ids=case.relevant_ids,
                checks=[CheckResult.from_outcome(o) for o in outcomes],
            )
        )

    return SuiteResult(
        name=suite.name,
        provider=provider.name,
        threshold=suite.threshold,
        cases=case_results,
    )
