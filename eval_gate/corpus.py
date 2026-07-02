"""Document corpus loading and a deterministic pure-Python TF-IDF retriever.

The retriever is intentionally dependency-free: it produces stable, reproducible
rankings so that retrieval metrics in deterministic mode never depend on a
vector database or network call. Swap in a production retriever behind the same
``Retriever.retrieve`` signature when wiring this into a real pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric word tokenization."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Document:
    """A single corpus document.

    Attributes:
        id: Stable document id (the filename stem).
        text: Full document text.
    """

    id: str
    text: str


class Corpus:
    """An in-memory collection of documents keyed by id."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs: dict[str, Document] = {}
        for d in docs:
            if d.id in self._docs:
                raise ValueError(f"duplicate document id in corpus: {d.id}")
            self._docs[d.id] = d

    def __len__(self) -> int:
        return len(self._docs)

    def ids(self) -> list[str]:
        """Sorted list of all document ids (deterministic order)."""
        return sorted(self._docs)

    def get(self, doc_id: str) -> Document | None:
        """Return a document by id, or ``None`` if absent."""
        return self._docs.get(doc_id)

    def contains_text(self, snippet: str) -> bool:
        """True if *snippet* (normalized) appears verbatim in any document."""
        norm = _normalize_ws(snippet)
        if not norm:
            return False
        return any(norm in _normalize_ws(d.text) for d in self._docs.values())

    def text_in(self, snippet: str, doc_id: str) -> bool:
        """True if *snippet* (normalized) appears in the given document."""
        doc = self._docs.get(doc_id)
        if doc is None:
            return False
        norm = _normalize_ws(snippet)
        return bool(norm) and norm in _normalize_ws(doc.text)


def _normalize_ws(text: str) -> str:
    return " ".join(text.lower().split())


def load_corpus(corpus_dir: str | Path) -> Corpus:
    """Load every ``*.md`` and ``*.txt`` file in *corpus_dir* as a document.

    The document id is the filename stem (e.g. ``refund_policy.md`` -> ``refund_policy``).

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the directory contains no documents.
    """
    d = Path(corpus_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {d}")

    docs: list[Document] = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in {".md", ".txt"} and f.is_file():
            docs.append(Document(id=f.stem, text=f.read_text(encoding="utf-8")))

    if not docs:
        raise ValueError(f"no .md/.txt documents found in corpus directory: {d}")
    return Corpus(docs)


class TfidfRetriever:
    """A deterministic TF-IDF retriever over a :class:`Corpus`.

    Ranking is by cosine similarity between the query and each document in
    TF-IDF space. Ties are broken by document id to keep results reproducible.
    """

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._doc_ids = corpus.ids()
        # Precompute document term frequencies and the IDF vector.
        self._doc_tf: dict[str, dict[str, float]] = {}
        df: dict[str, int] = {}
        for doc_id in self._doc_ids:
            doc = corpus.get(doc_id)
            assert doc is not None
            counts: dict[str, int] = {}
            for tok in tokenize(doc.text):
                counts[tok] = counts.get(tok, 0) + 1
            total = sum(counts.values()) or 1
            self._doc_tf[doc_id] = {t: c / total for t, c in counts.items()}
            for term in counts:
                df[term] = df.get(term, 0) + 1

        n = len(self._doc_ids) or 1
        # Smoothed IDF so common terms don't zero out and unseen terms are safe.
        self._idf: dict[str, float] = {
            term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()
        }

    def _vector(self, tf: dict[str, float]) -> dict[str, float]:
        return {t: freq * self._idf.get(t, math.log(1 + len(self._doc_ids)) + 1.0)
                for t, freq in tf.items()}

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Return the ids of the *top_k* most similar documents to *query*.

        Args:
            query: The query text.
            top_k: Number of ids to return (clamped to corpus size).

        Returns:
            Ordered list of document ids, best match first. Documents with zero
            similarity are excluded.
        """
        q_counts: dict[str, int] = {}
        for tok in tokenize(query):
            q_counts[tok] = q_counts.get(tok, 0) + 1
        q_total = sum(q_counts.values()) or 1
        q_tf = {t: c / q_total for t, c in q_counts.items()}
        q_vec = self._vector(q_tf)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))

        scored: list[tuple[float, str]] = []
        for doc_id in self._doc_ids:
            d_vec = self._vector(self._doc_tf[doc_id])
            dot = sum(q_vec.get(t, 0.0) * v for t, v in d_vec.items())
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
            sim = 0.0 if q_norm == 0 or d_norm == 0 else dot / (q_norm * d_norm)
            if sim > 0:
                scored.append((sim, doc_id))

        # Sort by descending similarity, then ascending id for stable ties.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for _, doc_id in scored[: max(0, top_k)]]
