"""Fallback handling utilities for Codex operations."""

from __future__ import annotations

import logging
from typing import Any

from in_memory_queue import InMemoryQueue
from llm_interface import OpenAIProvider, Prompt, LLMResult

# Single in-memory queue instance used for deferred prompts
_QUEUE = InMemoryQueue()


def queue_for_later(prompt: str | Prompt) -> None:
    """Queue *prompt* for later processing.

    The prompt text is stored in an :class:`InMemoryQueue` so that the request
    may be retried once normal operations resume.
    """

    text = prompt.user if isinstance(prompt, Prompt) else str(prompt)
    _QUEUE.send_task("codex_retry", {"prompt": text})


def reroute_to_lower_cost_model(prompt: str | Prompt) -> LLMResult:
    """Reroute *prompt* to a lower cost OpenAI model.

    Returns the completion from ``gpt-3.5-turbo``.
    """

    p = prompt if isinstance(prompt, Prompt) else Prompt(text=str(prompt))
    provider = OpenAIProvider(model="gpt-3.5-turbo")
    return provider.generate(p)


def handle_failure(prompt: str | Prompt, result: Any, reason: str) -> LLMResult | None:
    """Handle a failure from the primary Codex pipeline.

    The failure is logged and an attempt is made to reroute the *prompt* to a
    lower cost model.  If rerouting is unavailable the prompt is queued for
    later processing.
    """

    logger = logging.getLogger(__name__)
    logger.warning("Codex failure: %s", reason, extra={"result": result})

    try:
        return reroute_to_lower_cost_model(prompt)
    except Exception:
        logger.info("Reroute unavailable; queuing prompt for later")
        queue_for_later(prompt)
        return None


__all__ = [
    "handle_failure",
    "queue_for_later",
    "reroute_to_lower_cost_model",
]
