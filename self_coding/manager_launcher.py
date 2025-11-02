"""Utilities for launching the self-coding manager."""

from __future__ import annotations


def ensure_self_coding_ready() -> bool:
    """Return ``True`` when optional self-coding dependencies are importable."""

    try:
        import pydantic  # noqa: F401  # pragma: no cover - imported for side effects
        import sklearn  # noqa: F401  # pragma: no cover - imported for side effects
        import quick_fix_engine  # noqa: F401  # optional, but will block if referenced
        return True
    except ImportError as exc:  # pragma: no cover - executed only when deps missing
        print("Missing dependency for self-coding:", exc)
        return False
