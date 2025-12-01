from __future__ import annotations

"""Lightweight helpers for safe bootstrap access.

These wrappers centralize access to :func:`environment_bootstrap.ensure_bootstrapped`
without forcing heavy imports at module load time.
"""

from typing import Any


def bootstrap_state_snapshot() -> dict[str, bool]:
    """Return a lightweight snapshot of bootstrap readiness.

    The snapshot reflects the new bootstrap state API exposed by
    :mod:`environment_bootstrap` without triggering a bootstrap run.  Callers
    can use the ``ready`` and ``in_progress`` flags to decide whether they need
    to invoke :func:`ensure_bootstrapped` or defer until the current bootstrap
    completes.
    """

    try:
        from .environment_bootstrap import bootstrap_in_progress, is_bootstrapped
    except Exception:  # pragma: no cover - fallback for direct execution
        try:
            from environment_bootstrap import (  # type: ignore
                bootstrap_in_progress,
                is_bootstrapped,
            )
        except Exception:  # pragma: no cover - best effort snapshot
            return {"ready": False, "in_progress": False}

    ready = False
    in_progress = False
    try:
        ready = bool(is_bootstrapped())
    except Exception:  # pragma: no cover - conservative default
        ready = False
    try:
        in_progress = bool(bootstrap_in_progress())
    except Exception:  # pragma: no cover - conservative default
        in_progress = False

    return {"ready": ready, "in_progress": in_progress}


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


__all__ = [
    "bootstrap_state_snapshot",
    "ensure_bootstrapped",
    "ensure_environment_bootstrapped",
]
