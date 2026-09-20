"""Unit tests for generation-quality checks."""

from __future__ import annotations

import math

from eval_gate.checks import CheckContext, run_check
from eval_gate.checks.quality import faithfulness_score
from eval_gate.config import CheckSpec
from eval_gate.corpus import Corpus, Document


def _ctx(answer, retrieved=None, corpus=None):
    return CheckContext(
        answer=answer,
        retrieved_ids=retrieved or [],
        relevant_ids=[],
        corpus=corpus or Corpus([Document(id="d", text="")]),
    )


def test_length_bounds(tiny_corpus):
    ctx = _ctx("one two three four five", corpus=tiny_corpus)
    assert run_check(CheckSpec("length", {"min_words": 3, "max_words": 10}), ctx).passed
    assert not run_check(CheckSpec("length", {"max_words": 4}), ctx).passed
    assert not run_check(CheckSpec("length", {"min_words": 6}), ctx).passed


def test_forbidden_phrases(tiny_corpus):
    ctx = _ctx("As an AI language model, I think this is probably fine.", corpus=tiny_corpus)
    out = run_check(CheckSpec("forbidden_phrases", {"phrases": ["as an AI language model", "I think"]}), ctx)
    assert not out.passed
    assert len(out.detail["found"]) == 2
    clean = _ctx("Refunds are available within 30 days.", corpus=tiny_corpus)
    assert run_check(CheckSpec("forbidden_phrases", {"phrases": ["as an AI language model"]}), clean).passed


def test_format_json(tiny_corpus):
    assert run_check(CheckSpec("format", {"kind": "json"}), _ctx('{"a": 1}', corpus=tiny_corpus)).passed
    assert not run_check(CheckSpec("format", {"kind": "json"}), _ctx("not json", corpus=tiny_corpus)).passed


def test_format_markdown_list(tiny_corpus):
    md = _ctx("Methods:\n- app\n- key", corpus=tiny_corpus)
    assert run_check(CheckSpec("format", {"kind": "markdown_list"}), md).passed
    assert not run_check(CheckSpec("format", {"kind": "markdown_list"}), _ctx("just prose", corpus=tiny_corpus)).passed


def test_format_plain(tiny_corpus):
    assert run_check(CheckSpec("format", {"kind": "plain"}), _ctx("hello world", corpus=tiny_corpus)).passed
    assert not run_check(CheckSpec("format", {"kind": "plain"}), _ctx('{"a":1}', corpus=tiny_corpus)).passed


def test_faithfulness_score_math():
    context = "Refunds are available within 30 days of purchase."
    # every content word is present in context -> 1.0
    assert faithfulness_score("Refunds available within 30 days [doc:x]", context) == 1.0
    # a hallucinated content word lowers the score below 1.0
    score = faithfulness_score("Refunds available within 90 days via bitcoin", context)
    assert 0.0 < score < 1.0


def test_faithfulness_contentless_answer_scores_zero():
    """Changed 2026-09-19 (audit finding D4): this used to return 1.0, so an empty
    answer passed the faithfulness check. A non-answer is not a grounded answer."""
    assert faithfulness_score("the and of", "anything") == 0.0
    assert faithfulness_score("", "anything") == 0.0


def test_faithfulness_is_polarity_aware():
    """Audit finding D3: negation used to be stripped as a stopword."""
    context = "We do not sell personal data to third parties."
    assert faithfulness_score("We do not sell personal data to third parties.", context) == 1.0
    assert faithfulness_score("We sell personal data to third parties.", context) < 0.8


def test_faithfulness_check(tiny_corpus):
    corpus = Corpus([Document(id="alpha", text="Refunds are available within 30 days.")])
    ctx = CheckContext(
        answer="Refunds available within 30 days.",
        retrieved_ids=["alpha"],
        relevant_ids=[],
        corpus=corpus,
    )
    assert run_check(CheckSpec("faithfulness", {"min": 0.8}), ctx).passed
    bad = CheckContext(
        answer="You can pay with bitcoin and receive unicorn stickers.",
        retrieved_ids=["alpha"],
        relevant_ids=[],
        corpus=corpus,
    )
    assert not run_check(CheckSpec("faithfulness", {"min": 0.8}), bad).passed
