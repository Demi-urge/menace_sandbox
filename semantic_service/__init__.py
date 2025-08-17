"""Deprecated thin wrappers for :mod:`vector_service`.

This package now simply re-exports the public interface from
:mod:`vector_service` and emits deprecation warnings on import.
It will be removed once all consumers migrate to ``vector_service``.
"""

import warnings
from vector_service import (
    Retriever,
    FallbackResult,
    ContextBuilder,
    PatchLogger,
    EmbeddingBackfill,
    EmbeddableDBMixin,
    SemanticServiceError,
    VectorServiceError,
    RateLimitError,
    MalformedPromptError,
)

warnings.warn(
    "`semantic_service` is deprecated; use `vector_service` instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Retriever",
    "FallbackResult",
    "ContextBuilder",
    "PatchLogger",
    "EmbeddingBackfill",
    "EmbeddableDBMixin",
    "SemanticServiceError",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]
