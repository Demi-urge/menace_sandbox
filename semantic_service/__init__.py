"""Public interface for the :mod:`semantic_service` package.

The individual helper classes live in their own modules in order to keep the
implementations light‑weight and to avoid importing heavy dependencies at
startup.  This ``__init__`` simply re‑exports the commonly used classes so
callers can rely on ``semantic_service`` as a stable facade.
"""

from .retriever import Retriever
from .context_builder import ContextBuilder
from .patch_logger import PatchLogger
from .embedding_backfill import EmbeddingBackfill
from .exceptions import (
    SemanticServiceError,
    VectorServiceError,
    RateLimitError,
    MalformedPromptError,
)

try:  # pragma: no cover - optional dependency used in tests
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover - fallback when dependency missing
    EmbeddableDBMixin = object  # type: ignore


__all__ = [
    "Retriever",
    "ContextBuilder",
    "PatchLogger",
    "EmbeddingBackfill",
    "EmbeddableDBMixin",
    "SemanticServiceError",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]

