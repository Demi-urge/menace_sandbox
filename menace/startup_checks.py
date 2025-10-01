"""Compatibility wrapper exposing :mod:`startup_checks` under ``menace``."""
from __future__ import annotations

from startup_checks import *  # noqa: F401,F403

# Re-export ``__all__`` for ``from menace.startup_checks import *`` consumers.
try:
    from startup_checks import __all__ as __all__  # type: ignore
except ImportError:  # pragma: no cover - defensive fallback
    __all__ = []  # type: ignore[var-annotated]
