"""Retrieval and citation-grounding checks.

Pure IR metric functions (recall@k, precision@k, MRR, NDCG@k) plus checks that
gate on them, and a citation-grounding check that verifies the answer cites
documents that are (a) real corpus documents and (b) actually retrieved.
"""

from __future__ import annotations

import math
import re

from eval_gate.checks import (
    CheckContext,
    CheckOutcome,
    _bool_flag,
    _param,
    _positive_int,
    _unit_interval,
    register,
)
from eval_gate.config import CheckSpec


def _metric_params(spec: CheckSpec) -> list[str]:
    """``min`` must be a real threshold in [0, 1] and ``k`` a rank of at least 1."""
    return _unit_interval(spec, "min") + _positive_int(spec, "k")


# Matches citation markers like [doc:refund_policy] or [doc: shipping_policy].
_CITATION_RE = re.compile(r"\[doc:\s*([a-z0-9_\-]+)\s*\]", re.IGNORECASE)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant docs present in the top-k retrieved. 0.0 if no relevant."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant. 0.0 if k <= 0."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(top_k)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant doc. 0.0 if none present."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Binary-relevance NDCG@k. 0.0 if no relevant docs or k <= 0."""
    if k <= 0 or not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def extract_citations(text: str) -> list[str]:
    """Return lowercased doc ids cited via ``[doc:<id>]`` markers, in order."""
    return [m.group(1).lower() for m in _CITATION_RE.finditer(text)]


def _metric_check(spec: CheckSpec, ctx: CheckContext, value: float) -> CheckOutcome:
    min_value = float(_param(spec, "min", required=False, default=1.0))
    k = int(_param(spec, "k", required=False, default=len(ctx.retrieved_ids)))
    ok = value >= min_value - 1e-9
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=f"{spec.type}={value:.3f} (min {min_value:.3f}, k={k})",
        detail={"value": round(value, 6), "min": min_value, "k": k},
    )


@register("recall_at_k", params={"min", "k"}, validator=_metric_params)
def _recall(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    k = int(_param(spec, "k", required=False, default=len(ctx.retrieved_ids)))
    return _metric_check(spec, ctx, recall_at_k(ctx.retrieved_ids, ctx.relevant_ids, k))


@register("precision_at_k", params={"min", "k"}, validator=_metric_params)
def _precision(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    k = int(_param(spec, "k", required=False, default=len(ctx.retrieved_ids)))
    return _metric_check(spec, ctx, precision_at_k(ctx.retrieved_ids, ctx.relevant_ids, k))


@register("mrr", params={"min"}, validator=_metric_params)  # MRR has no cutoff
def _mrr(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    return _metric_check(spec, ctx, mrr(ctx.retrieved_ids, ctx.relevant_ids))


@register("ndcg_at_k", params={"min", "k"}, validator=_metric_params)
def _ndcg(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    k = int(_param(spec, "k", required=False, default=len(ctx.retrieved_ids)))
    return _metric_check(spec, ctx, ndcg_at_k(ctx.retrieved_ids, ctx.relevant_ids, k))


@register(
    "citations_grounded",
    params={"require_citation"},
    validator=lambda spec: _bool_flag(spec, "require_citation"),
)
def _citations_grounded(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    """Every ``[doc:<id>]`` cited must be a real corpus doc and be retrieved.

    Params:
        require_citation: If true (default), an answer with zero citations fails.
    """
    require_citation = bool(_param(spec, "require_citation", required=False, default=True))
    cited = extract_citations(ctx.answer)
    retrieved = set(ctx.retrieved_ids)

    unknown = [c for c in cited if ctx.corpus.get(c) is None]
    not_retrieved = [c for c in cited if ctx.corpus.get(c) is not None and c not in retrieved]

    if not cited:
        ok = not require_citation
        msg = "no citations present" + ("" if ok else " (a citation was required)")
        return CheckOutcome(
            check_type=spec.type,
            passed=ok,
            message=msg,
            detail={"cited": [], "unknown": [], "not_retrieved": []},
        )

    ok = not unknown and not not_retrieved
    if ok:
        msg = f"all {len(cited)} citation(s) are real and retrieved"
    else:
        problems = []
        if unknown:
            problems.append(f"unknown docs {unknown}")
        if not_retrieved:
            problems.append(f"cited-but-not-retrieved {not_retrieved}")
        msg = "; ".join(problems)
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=msg,
        detail={"cited": cited, "unknown": unknown, "not_retrieved": not_retrieved},
    )
