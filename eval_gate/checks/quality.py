"""Generation quality checks with explicit, deterministic rules.

Implements: ``length`` (word-count bounds), ``forbidden_phrases`` (banned
strings), ``format`` (must look like json / markdown_list / plain), and
``faithfulness`` (heuristic token-grounding of the answer against retrieved
context). These are rule-based, not an LLM-judge, so results are reproducible
and free. ``faithfulness`` is explicitly a lexical-overlap heuristic, not a
semantic entailment model — documented as such.
"""

from __future__ import annotations

import json
import re

from eval_gate.checks import CheckContext, CheckOutcome, _param, register
from eval_gate.checks.retrieval import _CITATION_RE
from eval_gate.config import CheckSpec
from eval_gate.corpus import tokenize

# Common English stopwords excluded from faithfulness token-overlap so the score
# reflects content words, not filler.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at for with without from by as is are was were
    be been being do does did have has had you your yours we our i me my it its this that these
    those they them their he she his her which who whom what when where why how not no can cannot
    will would should could may might must about into over under again more most some any all each
    """.split()
)


def _words(text: str) -> list[str]:
    return text.split()


@register("length")
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


@register("forbidden_phrases")
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


@register("format")
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


def faithfulness_score(answer: str, context_text: str) -> float:
    """Lexical-overlap faithfulness: fraction of the answer's content words that
    also appear in the retrieved context. Citation markers are stripped first.

    Returns 1.0 when the answer has no content words (nothing to hallucinate).
    """
    stripped = _CITATION_RE.sub(" ", answer)
    answer_terms = [t for t in tokenize(stripped) if t not in _STOPWORDS]
    if not answer_terms:
        return 1.0
    context_terms = set(tokenize(context_text))
    supported = sum(1 for t in answer_terms if t in context_terms)
    return supported / len(answer_terms)


@register("faithfulness")
def _faithfulness(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    min_value = float(_param(spec, "min", required=False, default=0.8))
    context_text = "\n".join(
        (ctx.corpus.get(doc_id).text if ctx.corpus.get(doc_id) else "")
        for doc_id in ctx.retrieved_ids
    )
    score = faithfulness_score(ctx.answer, context_text)
    ok = score >= min_value - 1e-9
    return CheckOutcome(
        check_type=spec.type,
        passed=ok,
        message=f"faithfulness={score:.3f} (min {min_value:.3f})",
        detail={"value": round(score, 6), "min": min_value},
    )
