"""Compatibility wrapper for :mod:`vector_service.context_builder`.

Import ``ContextBuilder`` from :mod:`vector_service` instead of this
module. This file re-exports the class and will be removed once
all consumers migrate.
"""

import warnings
from vector_service import ContextBuilder

__all__ = ["ContextBuilder"]

warnings.warn(
    "`semantic_service.context_builder` is deprecated; use `vector_service.ContextBuilder`",
    DeprecationWarning,
    stacklevel=2,
)
