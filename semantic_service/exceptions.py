"""Compatibility wrappers for :mod:`vector_service.exceptions`.

Import exceptions from :mod:`vector_service` instead. This module will be
removed once all consumers migrate.
"""

import warnings
from vector_service.exceptions import (
    SemanticServiceError,
    VectorServiceError,
    RateLimitError,
    MalformedPromptError,
)

__all__ = [
    "SemanticServiceError",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]

warnings.warn(
    "`semantic_service.exceptions` is deprecated; use `vector_service.exceptions`",
    DeprecationWarning,
    stacklevel=2,
)
