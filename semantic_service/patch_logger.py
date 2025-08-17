"""Compatibility wrapper for :mod:`vector_service.patch_logger`.

Use :class:`vector_service.PatchLogger` instead. This module will be removed
once all consumers migrate.
"""

import warnings
from vector_service import PatchLogger

__all__ = ["PatchLogger"]

warnings.warn(
    "`semantic_service.patch_logger` is deprecated; use `vector_service.PatchLogger`",
    DeprecationWarning,
    stacklevel=2,
)
