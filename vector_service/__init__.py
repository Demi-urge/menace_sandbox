"""Public interface for the :mod:`vector_service` package.

This package provides the canonical vector retrieval service.
"""

class _Stub:  # pragma: no cover - simple callable placeholder
    def __init__(self, *args, **kwargs):
        pass


# ``Retriever`` historically provided ``FallbackResult`` so we default to the
# lightweight implementations first and upgrade them when the real modules are
# available.  This ensures partial imports do not mask functionality unrelated to
# a missing heavy dependency (for example ``transformers``).
class FallbackResult(list):  # pragma: no cover - used when retriever unavailable
    pass


class VectorServiceError(Exception):  # pragma: no cover - default error type
    pass


RateLimitError = MalformedPromptError = VectorServiceError

Retriever = PatchLogger = CognitionLayer = EmbeddingBackfill = SharedVectorService = ContextBuilder = _Stub  # type: ignore

try:  # pragma: no cover - upgrade default errors when available
    from .exceptions import (
        VectorServiceError as _VectorServiceError,
        RateLimitError as _RateLimitError,
        MalformedPromptError as _MalformedPromptError,
    )
except Exception:
    pass
else:
    VectorServiceError = _VectorServiceError
    RateLimitError = _RateLimitError
    MalformedPromptError = _MalformedPromptError

try:  # pragma: no cover - optional heavy dependency
    from .retriever import Retriever as _Retriever, FallbackResult as _FallbackResult
except Exception:
    pass
else:
    Retriever = _Retriever
    FallbackResult = _FallbackResult

try:  # pragma: no cover - optional heavy dependency
    from .patch_logger import PatchLogger as _PatchLogger
except Exception:
    pass
else:
    PatchLogger = _PatchLogger

try:  # pragma: no cover - optional heavy dependency
    from .cognition_layer import CognitionLayer as _CognitionLayer
except Exception:
    pass
else:
    CognitionLayer = _CognitionLayer

try:  # pragma: no cover - optional heavy dependency
    from .embedding_backfill import EmbeddingBackfill as _EmbeddingBackfill
except Exception:
    pass
else:
    EmbeddingBackfill = _EmbeddingBackfill

try:  # pragma: no cover - optional heavy dependency
    from .vectorizer import SharedVectorService as _SharedVectorService
except Exception:
    pass
else:
    SharedVectorService = _SharedVectorService

try:  # pragma: no cover - optional heavy dependency
    from .context_builder import ContextBuilder as _ContextBuilder
except Exception:
    pass
else:
    ContextBuilder = _ContextBuilder


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
