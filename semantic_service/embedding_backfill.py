"""Compatibility wrapper for :mod:`vector_service.embedding_backfill`.

Use :class:`vector_service.EmbeddingBackfill` instead. This module will be
removed once all consumers migrate.
"""

import warnings
from vector_service import EmbeddingBackfill, EmbeddableDBMixin

__all__ = ["EmbeddingBackfill", "EmbeddableDBMixin"]

warnings.warn(
    "`semantic_service.embedding_backfill` is deprecated; use `vector_service.EmbeddingBackfill`",
    DeprecationWarning,
    stacklevel=2,
)
