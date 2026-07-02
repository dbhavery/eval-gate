"""Model output providers.

Two modes:

* **deterministic** (default) — reads recorded model outputs from a responses
  directory. No API keys, no network, byte-for-byte reproducible. This is what
  makes the CI gate stable.
* **provider** — calls a real LLM (OpenAI or Anthropic) using the retrieved
  corpus context. Enabled only when the relevant API key env var is set; the
  SDK is imported lazily so the package installs and runs without it.

A provider returns the model's answer text for a case. Retrieval itself is
always computed locally by :class:`~eval_gate.corpus.TfidfRetriever`, so only
the *generation* step differs between modes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from eval_gate.corpus import Corpus


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce an answer."""


class Provider(Protocol):
    """A source of model answers for evaluation cases."""

    name: str

    def answer(self, case_id: str, question: str, context_ids: list[str], corpus: Corpus) -> str:
        """Return the model's answer text for a case."""
        ...


class DeterministicProvider:
    """Serves recorded answers from ``<responses_dir>/<case_id>.json``.

    Each response file is a JSON object with at least an ``answer`` string::

        {"answer": "You may request a refund within 30 days ... [doc:refund_policy]"}

    Recording real outputs once and replaying them is what lets the gate run
    offline and stay reproducible across machines and CI.
    """

    name = "deterministic"

    def __init__(self, responses_dir: str | Path) -> None:
        self._dir = Path(responses_dir)
        if not self._dir.is_dir():
            raise ProviderError(f"responses directory not found: {self._dir}")

    def answer(self, case_id: str, question: str, context_ids: list[str], corpus: Corpus) -> str:
        f = self._dir / f"{case_id}.json"
        if not f.is_file():
            raise ProviderError(
                f"no recorded response for case '{case_id}' (expected {f}). "
                "Record it, or run with a live provider."
            )
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"malformed response fixture {f}: {exc}") from exc
        if not isinstance(data, dict) or "answer" not in data:
            raise ProviderError(f"response fixture {f} must contain an 'answer' field")
        return str(data["answer"])


def _build_context(context_ids: list[str], corpus: Corpus) -> str:
    parts = []
    for doc_id in context_ids:
        doc = corpus.get(doc_id)
        if doc is not None:
            parts.append(f"[doc:{doc_id}]\n{doc.text}")
    return "\n\n".join(parts)


_SYSTEM_PROMPT = (
    "You are a precise support assistant. Answer ONLY from the provided context. "
    "Cite the document you used with a marker like [doc:<id>]. If the context does "
    "not contain the answer, say you don't have that information and do not guess."
)


class OpenAIProvider:
    """Calls the OpenAI Chat Completions API. Requires ``OPENAI_API_KEY``."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def answer(self, case_id: str, question: str, context_ids: list[str], corpus: Corpus) -> str:
        try:
            from openai import OpenAI  # noqa: PLC0415 - lazy import, optional dep
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ProviderError(
                "openai package not installed. `pip install eval-gate[openai]`."
            ) from exc
        client = OpenAI()  # reads OPENAI_API_KEY from env
        context = _build_context(context_ids, corpus)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider:
    """Calls the Anthropic Messages API. Requires ``ANTHROPIC_API_KEY``."""

    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        self.model = model

    def answer(self, case_id: str, question: str, context_ids: list[str], corpus: Corpus) -> str:
        try:
            import anthropic  # noqa: PLC0415 - lazy import, optional dep
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ProviderError(
                "anthropic package not installed. `pip install eval-gate[anthropic]`."
            ) from exc
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        context = _build_context(context_ids, corpus)
        msg = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        # Concatenate text blocks.
        return "".join(getattr(b, "text", "") for b in msg.content)


def build_provider(mode: str, responses_dir: str | Path) -> Provider:
    """Construct a provider for *mode*.

    Args:
        mode: One of ``deterministic``, ``openai``, ``anthropic``, or ``auto``.
            ``auto`` picks a live provider if the matching API key is set,
            otherwise falls back to deterministic.
        responses_dir: Directory of recorded responses (deterministic mode).

    Returns:
        A ready-to-use :class:`Provider`.

    Raises:
        ProviderError: If a live provider is requested without its API key.
    """
    mode = mode.lower()
    if mode == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicProvider()
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        return DeterministicProvider(responses_dir)
    if mode == "deterministic":
        return DeterministicProvider(responses_dir)
    if mode == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set; cannot use the openai provider.")
        return OpenAIProvider()
    if mode == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set; cannot use the anthropic provider.")
        return AnthropicProvider()
    raise ProviderError(f"unknown provider mode: {mode!r}")
