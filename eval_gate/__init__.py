"""eval-gate — an evaluation release gate for LLM and RAG systems.

A small, dependency-light framework that evaluates LLM/RAG outputs against
graded fixtures and returns a non-zero exit code when quality regresses against
a recorded baseline. Runs fully offline in deterministic mode (no API keys).
"""

__version__ = "0.1.0"
