"""Compatibility wrapper for :mod:`vector_service.retriever`.

Import :class:`vector_service.Retriever` instead. This module will be removed
once all consumers migrate.
"""

import warnings
from vector_service import Retriever, FallbackResult

__all__ = ["Retriever", "FallbackResult"]

warnings.warn(
    "`semantic_service.retriever` is deprecated; use `vector_service.Retriever`",
    DeprecationWarning,
    stacklevel=2,
)
