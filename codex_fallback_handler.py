"""Simple fallback handler for Codex prompts.

This module provides minimal helpers used when Codex fails to produce a
response.  Failed prompts are persisted to a JSONL file so they can be replayed
later and, when possible, execution is rerouted to ``gpt-3.5-turbo``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - allow flat imports
    from .llm_interface import LLMClient, Prompt, LLMResult
except Exception:  # pragma: no cover - fallback for direct execution
    from llm_interface import LLMClient, Prompt, LLMResult  # type: ignore


# Location where failed prompts are stored for later replay
_QUEUE_FILE = Path("codex_fallback_queue.jsonl")

logger = logging.getLogger(__name__)


def queue_failed(prompt: Prompt, reason: str, *, path: Path = _QUEUE_FILE) -> None:
    """Persist ``prompt`` and ``reason`` as a JSONL record.

    Parameters
    ----------
    prompt:
        Prompt that triggered the failure.
    reason:
        Explanation of why the prompt could not be processed.
    path:
        Optional override for the queue location, mainly used in tests.
    """

    record = {"prompt": prompt.user, "reason": reason}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def reroute_to_gpt35(prompt: Prompt) -> LLMResult:
    """Retry ``prompt`` using ``gpt-3.5-turbo``.

    The helper now returns the full :class:`LLMResult` object so callers can
    inspect metadata such as token usage in addition to the generated text.
    """

    client = LLMClient(model="gpt-3.5-turbo")
    return client.generate(prompt)


def handle(
    prompt: Prompt, reason: str, *, queue_path: Optional[Path] = None
) -> LLMResult:
    """Attempt to reroute ``prompt`` and queue it on persistent failure.

    Parameters
    ----------
    prompt:
        The original prompt sent to Codex.
    reason:
        Explanation of why the fallback was triggered.
    queue_path:
        Optional override for the queue location, mainly used in tests.

    Returns
    -------
    LLMResult
        Result from :func:`reroute_to_gpt35`.  When rerouting fails or yields an
        empty completion, an ``LLMResult`` with an empty ``text`` field and the
        ``reason`` stored under ``raw`` is returned.
    """

    logger.warning("codex fallback invoked", extra={"reason": reason})

    try:
        result = reroute_to_gpt35(prompt)
    except Exception:
        logger.warning(
            "codex fallback reroute failed", exc_info=True, extra={"reason": reason}
        )
        queue_failed(prompt, reason, path=queue_path or _QUEUE_FILE)
        return LLMResult(text="", raw={"reason": reason})

    # Expose the routed model's text via ``result.text`` while preserving
    # provider specific metadata under ``result.raw``.
    if not getattr(result, "text", "").strip():
        logger.warning(
            "codex fallback produced no completion", extra={"reason": reason}
        )
        queue_failed(prompt, reason, path=queue_path or _QUEUE_FILE)
        return LLMResult(text="", raw={"reason": reason})
    return result


__all__ = ["queue_failed", "reroute_to_gpt35", "handle"]

