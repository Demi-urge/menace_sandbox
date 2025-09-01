"""Orphan handling helpers for the self-improvement engine.

Functions in this module wrap optional ``sandbox_runner`` hooks used to
integrate and reclassify orphaned modules.  The wrappers provide consistent
error handling and retry semantics so callers receive actionable failures when
configuration is incomplete.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .utils import _call_with_retries


def _load_orphan_module(attr: str) -> Callable[..., Any]:
    try:  # pragma: no cover - optional dependency
        from sandbox_runner import orphan_integration as _oi
    except Exception as exc:  # pragma: no cover - exercised when optional
        raise RuntimeError(
            "sandbox_runner is required for orphan integration."
            " Install the sandbox runner package or add it to PYTHONPATH."
        ) from exc
    if not hasattr(_oi, attr):  # pragma: no cover - missing attribute
        raise RuntimeError(
            f"sandbox_runner.orphan_integration.{attr} is missing;"
            " ensure the sandbox runner is up to date."
        )
    return getattr(_oi, attr)


def integrate_orphans(
    *args: object, retries: int = 3, delay: float = 0.1, **kwargs: object
) -> list[str]:
    """Invoke sandbox runner orphan integration with safeguards."""
    func = _load_orphan_module("integrate_orphans")
    return _call_with_retries(func, *args, retries=retries, delay=delay, **kwargs)


def post_round_orphan_scan(
    *args: object, retries: int = 3, delay: float = 0.1, **kwargs: object
) -> Dict[str, object]:
    """Trigger the sandbox post-round orphan scan."""
    func = _load_orphan_module("post_round_orphan_scan")
    return _call_with_retries(func, *args, retries=retries, delay=delay, **kwargs)


__all__ = ["integrate_orphans", "post_round_orphan_scan"]
