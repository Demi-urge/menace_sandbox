from __future__ import annotations

"""Lightweight helpers for safe bootstrap access.

These wrappers centralize access to :func:`environment_bootstrap.ensure_bootstrapped`
without forcing heavy imports at module load time.
"""

from typing import Any


def ensure_bootstrapped(**kwargs: Any) -> dict[str, object]:
    """Ensure environment bootstrap runs once in a process-safe manner.

    The import is deferred to avoid circular imports for modules that are
    pulled in early during bootstrap orchestration. Keep this helper as the
    single entry point so constructors and module-level initializers never
    spawn ad-hoc bootstrap routines.
    """

    try:
        from .environment_bootstrap import ensure_bootstrapped as _ensure_bootstrapped
    except Exception:  # pragma: no cover - fallback for direct execution
        from environment_bootstrap import ensure_bootstrapped as _ensure_bootstrapped  # type: ignore
    return _ensure_bootstrapped(**kwargs)


def ensure_environment_bootstrapped(**kwargs: Any) -> dict[str, object]:
    """Backwards-compatible alias for :func:`ensure_bootstrapped`."""

    return ensure_bootstrapped(**kwargs)


__all__ = ["ensure_bootstrapped", "ensure_environment_bootstrapped"]
