"""Shared test fixtures and path helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"

# Ensure the package is importable when running pytest from a fresh checkout
# without an editable install.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_gate.corpus import Corpus, Document  # noqa: E402


@pytest.fixture
def tiny_corpus() -> Corpus:
    return Corpus(
        [
            Document(id="alpha", text="Refunds are available within 30 days of purchase."),
            Document(id="beta", text="Shipping takes 3 to 5 business days in the US."),
            Document(id="gamma", text="The warranty covers manufacturing defects for 24 months."),
        ]
    )
