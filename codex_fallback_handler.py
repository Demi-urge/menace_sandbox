"""Fallback handling for Codex operations.

This module provides helper utilities to either queue failed prompts for
later retry or reroute them to a lower cost model.  The behaviour is controlled
via the ``CODEX_FALLBACK_STRATEGY`` environment variable which accepts either
``"queue"`` or ``"reroute"``.  Individual calls to :func:`handle_failure` may
override this by passing ``strategy`` as a keyword argument.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dynamic_path_router import resolve_path
from llm_interface import LLMResult, OpenAIProvider, Prompt

try:  # pragma: no cover - metrics are optional
    from .metrics_exporter import Gauge  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    try:
        from metrics_exporter import Gauge  # type: ignore
    except Exception:  # pragma: no cover - metrics unavailable
        Gauge = None  # type: ignore


logger = logging.getLogger(__name__)

# Default location for the retry queue stored as JSONL
_QUEUE_FILE = resolve_path(os.getenv("CODEX_RETRY_QUEUE", "codex_retry_queue.jsonl"))
# Default strategy for handling failures: "queue" or "reroute"
_DEFAULT_STRATEGY = os.getenv("CODEX_FALLBACK_STRATEGY", "reroute").lower()

# Optional gauges for tracking behaviour
_QUEUE_COUNT = (
    Gauge("codex_retry_queue_total", "Prompts queued for Codex retry")
    if Gauge
    else None
)
_REROUTE_COUNT = (
    Gauge("codex_reroute_total", "Prompts rerouted to alternate model")
    if Gauge
    else None
)
_REROUTE_FAILURES = (
    Gauge("codex_reroute_failures_total", "Reroute attempts that failed")
    if Gauge
    else None
)


def queue_for_retry(prompt: str | Prompt, *, path: Path = _QUEUE_FILE) -> None:
    """Append *prompt* to a disk-backed retry queue."""

    text = prompt.user if isinstance(prompt, Prompt) else str(prompt)
    record = {"prompt": text}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    logger.info("queued prompt for retry", extra={"prompt": text, "queue": str(path)})
    if _QUEUE_COUNT:
        _QUEUE_COUNT.inc()


def route_to_alt_model(prompt: str | Prompt, model: str = "gpt-3.5-turbo") -> LLMResult:
    """Rerun *prompt* using an alternate model."""

    p = prompt if isinstance(prompt, Prompt) else Prompt(str(prompt))
    try:  # Prefer router configuration if available
        from llm_router import client_from_settings

        client = client_from_settings()
    except Exception:  # Fall back to direct OpenAI access
        client = OpenAIProvider(model=model)
    logger.info(
        "rerouting prompt to alternate model",
        extra={"model": getattr(client, "model", model)},
    )
    if _REROUTE_COUNT:
        _REROUTE_COUNT.inc()
    return client.generate(p)


def handle_failure(
    prompt: str | Prompt,
    exc: Any | None = None,
    result: LLMResult | None = None,
    *,
    strategy: str | None = None,
) -> LLMResult | None:
    """Handle a Codex failure for *prompt*.

    If ``strategy`` (or ``CODEX_FALLBACK_STRATEGY``) is ``"queue"`` the prompt is
    appended to the retry queue.  Otherwise the prompt is rerouted to an
    alternate model.  Should rerouting fail the prompt is queued.
    """

    logger.warning(
        "codex failure", extra={"exception": exc, "result": getattr(result, "raw", result)}
    )
    mode = (strategy or _DEFAULT_STRATEGY).lower()
    if mode == "queue":
        queue_for_retry(prompt)
        return None
    try:
        return route_to_alt_model(prompt)
    except Exception as reroute_exc:  # pragma: no cover - network failure
        logger.error("reroute failed; queueing prompt", exc_info=reroute_exc)
        if _REROUTE_FAILURES:
            _REROUTE_FAILURES.inc()
        queue_for_retry(prompt)
        return None


__all__ = ["handle_failure", "queue_for_retry", "route_to_alt_model"]
