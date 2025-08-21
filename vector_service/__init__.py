"""Public interface for the :mod:`vector_service` package.

This package provides the canonical vector retrieval service.
"""

from .retriever import Retriever, FallbackResult
from .context_builder import ContextBuilder
from .patch_logger import PatchLogger
from .cognition_layer import CognitionLayer
from .embedding_backfill import EmbeddingBackfill
from .vectorizer import SharedVectorService
from .exceptions import (
    VectorServiceError,
    RateLimitError,
    MalformedPromptError,
)


class ErrorResult(Exception):
    """Fallback error result used when retriever returns an error."""

    pass

try:  # pragma: no cover - optional dependency used in tests
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover - fallback when dependency missing
    EmbeddableDBMixin = object  # type: ignore


__all__ = [
    "Retriever",
    "FallbackResult",
    "ContextBuilder",
    "PatchLogger",
    "CognitionLayer",
    "EmbeddingBackfill",
    "SharedVectorService",
    "EmbeddableDBMixin",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
    "ErrorResult",
]
