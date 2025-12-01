from __future__ import annotations

"""Lightweight helpers for safe bootstrap access.

These wrappers centralize access to :func:`environment_bootstrap.ensure_bootstrapped`
without forcing heavy imports at module load time.
"""

from typing import Any


def ensure_environment_bootstrapped(**kwargs: Any) -> bool:
    """Ensure environment bootstrap runs once in a process-safe manner.

    The import is deferred to avoid circular imports for modules that are
    pulled in early during bootstrap orchestration.
    """

    try:
        from .environment_bootstrap import ensure_bootstrapped
    except Exception:  # pragma: no cover - fallback for direct execution
        from environment_bootstrap import ensure_bootstrapped  # type: ignore
    return ensure_bootstrapped(**kwargs)


__all__ = ["ensure_environment_bootstrapped"]
