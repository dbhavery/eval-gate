"""Generation quality checks with explicit, deterministic rules.

Implements: ``length`` (word-count bounds), ``forbidden_phrases`` (banned
strings), ``format`` (must look like json / markdown_list / plain), and
``faithfulness`` (heuristic token-grounding of the answer against retrieved
context). These are rule-based, not an LLM-judge, so results are reproducible
and free. ``faithfulness`` is explicitly a *polarity-aware* lexical-overlap
heuristic. It is not a semantic entailment model, and says so.
"""

from __future__ import annotations

import json
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
from eval_gate.checks.retrieval import _CITATION_RE
from eval_gate.config import CheckSpec
from eval_gate.corpus import tokenize

# Common English stopwords excluded from faithfulness token-overlap so the score
# reflects content words, not filler.
#
# NOTE (2026-09-19 audit, finding D3): negation words are deliberately NOT in
# this list. They used to be, which made "We do not sell personal data" and
# "We sell personal data" both score 1.000 against the same context. Anything
# that flips the meaning of a sentence is a content word here.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at for from by as is are was were
    be been being do does did have has had you your yours we our i me my it its this that these
    those they them their he she his her which who whom what when where why how can
    will would should could may might must about into over under again more most some any all each
    """.split()
)

# Words that negate the clause they open. A content word that follows one of
# these inside the same clause is recorded with the opposite polarity, so a
# dropped or invented "not" changes the score instead of vanishing.
_NEGATION_CUES = frozenset(
    """
    not no never none neither nor nothing nobody nowhere without cannot cant wont
    unable lacks lack excluding
    """.split()
)

# "don't" / "isn't" / "won't" -> " do not" / " is not" / " will not" (the stem is
# a stopword either way; what matters is that the cue survives tokenization).
_CONTRACTED_NOT_RE = re.compile(r"n['’]t\b", re.IGNORECASE)

# A negation's scope ends at the next clause boundary. Splitting on "but" as
# well keeps "we do not X, but we do Y" from tagging Y as negated. A single
# newline is *not* a boundary: corpus documents are hard-wrapped, and treating a
# wrap as a boundary cut "we do not sell personal data / to third parties" in two
# and lost the negation on the second half.
_CLAUSE_SPLIT_RE = re.compile(r"[.;:!?,]+|\bbut\b", re.IGNORECASE)
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n\s*")


def _words(text: str) -> list[str]:
    return text.split()


def _length_bounds(spec: CheckSpec) -> list[str]:
    """A length check with no bound cannot fail, so it is refused."""
    errors = _positive_int(spec, "min_words") + _positive_int(spec, "max_words")
    if "min_words" not in spec.params and "max_words" not in spec.params:
        errors.append(
            "check 'length' needs at least one of 'min_words' or 'max_words'; "
            "with neither it can never fail"
        )
    lo, hi = spec.params.get("min_words"), spec.params.get("max_words")
    if isinstance(lo, int) and isinstance(hi, int) and lo > hi:
        errors.append(f"check 'length': min_words {lo} is above max_words {hi}")
    return errors


@register("length", params={"min_words", "max_words"}, validator=_length_bounds)
def _length(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    n = len(_words(ctx.answer))
    min_words = _param(spec, "min_words", required=False, default=None)
    max_words = _param(spec, "max_words", required=False, default=None)
    problems = []
    if min_words is not None and n < int(min_words):
        problems.append(f"{n} words < min {min_words}")
    if max_words is not None and n > int(max_words):
        problems.append(f"{n} words > max {max_words}")
    ok = not problems
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=(f"length {n} words within bounds" if ok else "; ".join(problems)),
        detail={"words": n, "min_words": min_words, "max_words": max_words},
    )


def _phrase_list(spec: CheckSpec) -> list[str]:
    phrases = spec.params.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        return [
            "check 'forbidden_phrases': 'phrases' must be a non-empty list "
            f"(got {phrases!r}); an empty list forbids nothing and cannot fail"
        ]
    bad = [p for p in phrases if not isinstance(p, str) or not p.strip()]
    if bad:
        return [f"check 'forbidden_phrases': every phrase must be a non-empty string, got {bad!r}"]
    return _bool_flag(spec, "case_insensitive")


@register(
    "forbidden_phrases",
    params={"phrases", "case_insensitive"},
    required={"phrases"},
    validator=_phrase_list,
)
def _forbidden(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    phrases = _param(spec, "phrases")
    if not isinstance(phrases, list):
        raise KeyError("check 'forbidden_phrases' requires a list 'phrases' parameter")
    ci = bool(_param(spec, "case_insensitive", required=False, default=True))
    hay = ctx.answer.lower() if ci else ctx.answer
    found = [p for p in phrases if (str(p).lower() if ci else str(p)) in hay]
    ok = not found
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=("no forbidden phrases present" if ok else f"found forbidden: {found}"),
        detail={"found": found},
    )


_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+\S", re.MULTILINE)


_FORMAT_KINDS = ("json", "markdown_list", "plain")


def _format_kind(spec: CheckSpec) -> list[str]:
    kind = spec.params.get("kind")
    if kind not in _FORMAT_KINDS:
        return [f"check 'format': 'kind' must be one of {list(_FORMAT_KINDS)}, got {kind!r}"]
    return []


@register("format", params={"kind"}, required={"kind"}, validator=_format_kind)
def _format(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    kind = str(_param(spec, "kind"))
    text = ctx.answer.strip()
    if kind == "json":
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        try:
            json.loads(candidate)
            ok, why = True, "parses as JSON"
        except json.JSONDecodeError as exc:
            ok, why = False, f"not JSON: {exc}"
    elif kind == "markdown_list":
        ok = bool(_MD_LIST_RE.search(text))
        why = "contains a markdown list" if ok else "no markdown list found"
    elif kind == "plain":
        ok = not text.startswith("{") and not text.startswith("```")
        why = "plain text" if ok else "looks like code/JSON, expected plain text"
    else:
        raise KeyError(f"unknown format kind '{kind}' (json|markdown_list|plain)")
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=f"format {kind}: {why}",
        detail={"kind": kind},
    )


def polarity_terms(text: str) -> list[tuple[str, bool]]:
    """Content words of *text* paired with the polarity of their clause.

    ``("sell", True)`` means the word appeared inside the scope of a negation
    ("we do not **sell**"); ``("sell", False)`` means it did not. Scope runs
    from the negation cue to the end of its clause.
    """
    terms: list[tuple[str, bool]] = []
    expanded = _CONTRACTED_NOT_RE.sub(" not", text)
    # A paragraph break ends a clause; a soft wrap inside a paragraph does not.
    expanded = _PARAGRAPH_BREAK_RE.sub(" . ", expanded).replace("\n", " ")
    for clause in _CLAUSE_SPLIT_RE.split(expanded):
        negated = False
        for tok in tokenize(clause):
            if tok in _NEGATION_CUES:
                negated = True
                continue
            if tok in _STOPWORDS:
                continue
            terms.append((tok, negated))
    return terms


def faithfulness_score(answer: str, context_text: str) -> float:
    """Polarity-aware lexical-overlap faithfulness.

    The fraction of the answer's content words that appear in the retrieved
    context *with the same polarity*. Citation markers are stripped first.

    Returns 0.0 when the answer has no content words: an empty or filler-only
    answer is not a grounded answer, it is a non-answer, and the gate must see
    it (2026-09-19 audit, finding D4; this used to return 1.0).
    """
    stripped = _CITATION_RE.sub(" ", answer)
    answer_terms = polarity_terms(stripped)
    if not answer_terms:
        return 0.0
    context_terms = set(polarity_terms(context_text))
    supported = sum(1 for pair in answer_terms if pair in context_terms)
    return supported / len(answer_terms)


@register(
    "faithfulness",
    params={"min"},
    validator=lambda spec: _unit_interval(spec, "min"),
)
def _faithfulness(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    min_value = float(_param(spec, "min", required=False, default=0.8))
    context_text = "\n".join(
        (ctx.corpus.get(doc_id).text if ctx.corpus.get(doc_id) else "")
        for doc_id in ctx.retrieved_ids
    )
    score = faithfulness_score(ctx.answer, context_text)
    ok = score >= min_value - 1e-9
    empty = not polarity_terms(_CITATION_RE.sub(" ", ctx.answer))
    note = " - answer has no content words" if empty else ""
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=f"faithfulness={score:.3f} (min {min_value:.3f}){note}",
        detail={"value": round(score, 6), "min": min_value, "contentless": empty},
    )
