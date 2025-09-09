"""Public interface for the :mod:`vector_service` package.

This package provides the canonical vector retrieval service.
"""

try:  # pragma: no cover - optional heavy dependencies
    from .retriever import Retriever, FallbackResult
    from .patch_logger import PatchLogger
    from .cognition_layer import CognitionLayer
    from .embedding_backfill import EmbeddingBackfill
    from .vectorizer import SharedVectorService
    from .context_builder import ContextBuilder
    from .exceptions import (
        VectorServiceError,
        RateLimitError,
        MalformedPromptError,
    )
except Exception:  # pragma: no cover - lightweight fallbacks for tests
    class _Stub:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    Retriever = PatchLogger = CognitionLayer = EmbeddingBackfill = SharedVectorService = ContextBuilder = _Stub  # type: ignore

    class FallbackResult(list):
        pass

    class VectorServiceError(Exception):
        pass

    RateLimitError = MalformedPromptError = VectorServiceError


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
    "PatchLogger",
    "CognitionLayer",
    "EmbeddingBackfill",
    "SharedVectorService",
    "ContextBuilder",
    "EmbeddableDBMixin",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
    "ErrorResult",
]
