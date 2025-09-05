"""Simple fallback handler for Codex prompts.

This module provides minimal helpers used when Codex fails to produce a
response.  Failed prompts are persisted to a JSONL file so they can be replayed
later and, when possible, execution is rerouted to ``gpt-3.5-turbo``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - allow flat imports
    from .llm_interface import LLMClient, Prompt, LLMResult
except Exception:  # pragma: no cover - fallback for direct execution
    from llm_interface import LLMClient, Prompt, LLMResult  # type: ignore


# Location where failed prompts are stored for later replay
_QUEUE_FILE = Path("codex_fallback_queue.jsonl")


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
) -> Optional[LLMResult]:
    """Attempt to reroute ``prompt`` and queue it on persistent failure.

    On success the :class:`LLMResult` from :func:`reroute_to_gpt35` is returned.
    If rerouting raises an exception, the prompt and failure ``reason`` are
    written to the queue and ``None`` is returned.
    """

    try:
        return reroute_to_gpt35(prompt)
    except Exception:
        queue_failed(prompt, reason, path=queue_path or _QUEUE_FILE)
        return None


__all__ = ["queue_failed", "reroute_to_gpt35", "handle"]

