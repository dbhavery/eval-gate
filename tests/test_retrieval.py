"""Unit tests for retrieval metric math, the retriever, and grounding checks."""

from __future__ import annotations

import math

from eval_gate.checks import CheckContext, run_check
from eval_gate.checks.retrieval import (
    extract_citations,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from eval_gate.config import CheckSpec
from eval_gate.corpus import TfidfRetriever


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == 0.5
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert recall_at_k(["x"], ["a"], 1) == 0.0
    assert recall_at_k(["a"], [], 1) == 0.0  # no relevant -> 0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a", "b"], 3) == 2 / 3
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert precision_at_k([], ["a"], 3) == 0.0
    assert precision_at_k(["a"], ["a"], 0) == 0.0


def test_mrr():
    assert mrr(["x", "a", "b"], ["a"]) == 0.5
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_at_k():
    # perfect ranking -> 1.0
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    # one relevant at rank 2: dcg=1/log2(3); idcg=1 -> value < 1
    val = ndcg_at_k(["x", "a"], ["a"], 2)
    assert math.isclose(val, 1.0 / math.log2(3), rel_tol=1e-9)
    assert ndcg_at_k(["x"], [], 2) == 0.0


def test_extract_citations():
    text = "See [doc:refund_policy] and [doc: shipping_policy] for details."
    assert extract_citations(text) == ["refund_policy", "shipping_policy"]
    assert extract_citations("no citations here") == []


def test_retriever_finds_relevant_doc(tiny_corpus):
    r = TfidfRetriever(tiny_corpus)
    top = r.retrieve("when are refunds available after purchase", top_k=1)
    assert top == ["alpha"]
    top_w = r.retrieve("warranty defects coverage", top_k=1)
    assert top_w == ["gamma"]


def test_retriever_empty_on_no_overlap(tiny_corpus):
    r = TfidfRetriever(tiny_corpus)
    assert r.retrieve("xyzzy nonexistent terms", top_k=3) == []


def test_retriever_is_deterministic(tiny_corpus):
    r = TfidfRetriever(tiny_corpus)
    q = "shipping business days"
    assert r.retrieve(q, top_k=3) == r.retrieve(q, top_k=3)


def _ctx(answer, retrieved, corpus):
    return CheckContext(answer=answer, retrieved_ids=retrieved, relevant_ids=[], corpus=corpus)


def test_citations_grounded_pass(tiny_corpus):
    ctx = _ctx("Refunds within 30 days. [doc:alpha]", ["alpha", "beta"], tiny_corpus)
    assert run_check(CheckSpec("citations_grounded", {}), ctx).passed


def test_citations_grounded_unknown_doc_fails(tiny_corpus):
    ctx = _ctx("See [doc:nonexistent].", ["alpha"], tiny_corpus)
    out = run_check(CheckSpec("citations_grounded", {}), ctx)
    assert not out.passed
    assert out.detail["unknown"] == ["nonexistent"]


def test_citations_grounded_not_retrieved_fails(tiny_corpus):
    # gamma is a real doc but was not retrieved for this case
    ctx = _ctx("See [doc:gamma].", ["alpha", "beta"], tiny_corpus)
    out = run_check(CheckSpec("citations_grounded", {}), ctx)
    assert not out.passed
    assert out.detail["not_retrieved"] == ["gamma"]


def test_citations_grounded_missing_citation_fails_when_required(tiny_corpus):
    ctx = _ctx("An answer with no citation.", ["alpha"], tiny_corpus)
    assert not run_check(CheckSpec("citations_grounded", {"require_citation": True}), ctx).passed
    assert run_check(CheckSpec("citations_grounded", {"require_citation": False}), ctx).passed


def test_recall_check_via_registry(tiny_corpus):
    ctx = CheckContext(answer="", retrieved_ids=["alpha", "beta"], relevant_ids=["alpha"], corpus=tiny_corpus)
    assert run_check(CheckSpec("recall_at_k", {"k": 2, "min": 1.0}), ctx).passed
    ctx2 = CheckContext(answer="", retrieved_ids=["beta"], relevant_ids=["alpha"], corpus=tiny_corpus)
    assert not run_check(CheckSpec("recall_at_k", {"k": 1, "min": 1.0}), ctx2).passed
