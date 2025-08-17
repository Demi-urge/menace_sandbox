"""Public interface for the :mod:`vector_service` package.

This package provides the canonical vector retrieval service.
"""

from .retriever import Retriever, FallbackResult
from .context_builder import ContextBuilder
from .patch_logger import PatchLogger
from .embedding_backfill import EmbeddingBackfill
from .exceptions import (
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
    "FallbackResult",
    "ContextBuilder",
    "PatchLogger",
    "EmbeddingBackfill",
    "EmbeddableDBMixin",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]
