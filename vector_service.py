"""Deprecated wrapper re-exporting :mod:`semantic_service` symbols.

The project consolidated `vector_service` and `semantic_service` into a single
canonical package.  Import from :mod:`semantic_service` directly; this module
exists only for backwards compatibility and will be removed in a future
release.
"""

from semantic_service import *  # noqa: F401,F403
import warnings

warnings.warn(
    "`vector_service` is deprecated; use `semantic_service` instead",
    DeprecationWarning,
    stacklevel=2,
)
