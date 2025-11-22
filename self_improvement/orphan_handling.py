"""Orphan handling helpers for the self-improvement engine.

Functions in this module wrap optional ``sandbox_runner`` hooks used to
integrate and reclassify orphaned modules.  The wrappers provide consistent
error handling and retry semantics so callers receive actionable failures when
configuration is incomplete.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import logging

from sandbox_settings import SandboxSettings

from .utils import _call_with_retries

from metrics_exporter import (
    orphan_integration_success_total,
    orphan_integration_failure_total,
)

logger = logging.getLogger(__name__)


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
    *args: object,
    retries: int | None = None,
    delay: float | None = None,
    create_default_context_builder: bool = True,
    **kwargs: object,
) -> list[str]:
    """Invoke sandbox runner orphan integration with safeguards."""
    settings = SandboxSettings()
    retries = retries if retries is not None else settings.orphan_retry_attempts
    delay = delay if delay is not None else settings.orphan_retry_delay
    repo = kwargs.get("repo")
    if repo is None and not args:
        repo = Path(settings.sandbox_repo_path)
    if repo is not None:
        kwargs.setdefault("repo", Path(repo))
    if create_default_context_builder and "context_builder" not in kwargs:
        from context_builder_util import create_context_builder

        kwargs["context_builder"] = create_context_builder()
    func = _load_orphan_module("integrate_orphans")
    try:
        modules = _call_with_retries(
            func, *args, retries=retries, delay=delay, **kwargs
        )
    except Exception:
        orphan_integration_failure_total.inc()
        logger.exception("orphan integration failed")
        raise
    else:
        orphan_integration_success_total.inc()
        logger.info("integrated modules: %s", modules)
        return modules


def post_round_orphan_scan(
    *args: object,
    retries: int | None = None,
    delay: float | None = None,
    create_default_context_builder: bool = True,
    **kwargs: object,
) -> Dict[str, object]:
    """Trigger the sandbox post-round orphan scan."""
    settings = SandboxSettings()
    retries = retries if retries is not None else settings.orphan_retry_attempts
    delay = delay if delay is not None else settings.orphan_retry_delay
    repo = kwargs.get("repo")
    if repo is None and not args:
        repo = Path(settings.sandbox_repo_path)
    if repo is not None:
        kwargs.setdefault("repo", Path(repo))
    if create_default_context_builder and "context_builder" not in kwargs:
        from context_builder_util import create_context_builder

        kwargs["context_builder"] = create_context_builder()
    func = _load_orphan_module("post_round_orphan_scan")
    try:
        result = _call_with_retries(
            func, *args, retries=retries, delay=delay, **kwargs
        )
    except Exception:
        orphan_integration_failure_total.inc()
        logger.exception("post round orphan scan failed")
        raise
    else:
        orphan_integration_success_total.inc()
        integrated = result.get("integrated") if isinstance(result, dict) else None
        flagged = result.get("flagged") if isinstance(result, dict) else None
        logger.info(
            "post round scan integrated=%s flagged=%s", integrated, flagged
        )
        return result


__all__ = ["integrate_orphans", "post_round_orphan_scan"]
