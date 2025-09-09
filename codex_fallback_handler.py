"""Simple fallback handler for Codex prompts.

This module provides minimal helpers used when Codex fails to produce a
response.  Failed prompts are persisted to a JSONL file so they can be replayed
later and, when possible, execution is rerouted to a configurable fallback
model (``gpt-3.5-turbo`` by default).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Any

try:  # pragma: no cover - allow flat imports
    from .llm_interface import LLMClient, Prompt, LLMResult
except Exception:  # pragma: no cover - fallback for direct execution
    from llm_interface import LLMClient, Prompt, LLMResult  # type: ignore

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback when vector service missing
    ContextBuilder = Any  # type: ignore

from sandbox_settings import SandboxSettings


_settings = SandboxSettings()


def _default_queue_path() -> Path:
    """Return the configured queue path.

    The value is pulled from :class:`SandboxSettings` via
    ``codex_retry_queue_path`` so tests can override the destination.
    """

    return Path(getattr(_settings, "codex_retry_queue_path", "codex_retry_queue.jsonl"))

logger = logging.getLogger(__name__)


def queue_failed(prompt: Prompt, reason: str, *, path: Optional[Path] = None) -> None:
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
    path = path or _default_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class _ContextClient:
    """Lightweight wrapper injecting vector-based context into prompts."""

    def __init__(self, *, model: str, context_builder: ContextBuilder) -> None:
        self._client = LLMClient(model=model)
        self._builder = context_builder

    def generate(self, prompt: Prompt) -> LLMResult:
        context = ""
        try:
            ctx_res = self._builder.build(prompt.user)
            context = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
        except Exception:
            context = ""
        if context:
            prompt = Prompt(
                f"{context}\n\n{prompt.user}",
                system=prompt.system,
                examples=prompt.examples,
                vector_confidence=prompt.vector_confidence,
                tags=prompt.tags,
                metadata=prompt.metadata,
            )
        return self._client.generate(prompt, context_builder=self._builder)


def reroute_to_fallback_model(prompt: Prompt, *, context_builder: ContextBuilder) -> LLMResult:
    """Retry ``prompt`` using the configured fallback model.

    The model is taken from :class:`SandboxSettings` via the
    ``codex_fallback_model`` attribute.  The helper returns the full
    :class:`LLMResult` object so callers can inspect metadata such as token
    usage in addition to the generated text.
    """

    model = getattr(_settings, "codex_fallback_model", "gpt-3.5-turbo")
    client = _ContextClient(model=model, context_builder=context_builder)
    return client.generate(prompt)


def handle(
    prompt: Prompt,
    reason: str,
    *,
    context_builder: ContextBuilder,
    queue_path: Optional[Path] = None,
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
        Result from :func:`reroute_to_fallback_model`.  When rerouting fails or yields an
        empty completion, an ``LLMResult`` with an empty ``text`` field and the
        ``reason`` stored under ``raw`` is returned.
    """

    logger.warning("codex fallback invoked", extra={"reason": reason})

    strategy = getattr(_settings, "codex_fallback_strategy", "queue")
    if strategy != "reroute":
        queue_failed(prompt, reason, path=queue_path)
        return LLMResult(text="", raw={"reason": reason})

    try:
        try:
            context_builder.refresh_db_weights()
        except Exception:
            pass
        try:
            context_builder.build(prompt.user)
        except Exception:
            pass
        result = reroute_to_fallback_model(prompt, context_builder=context_builder)
    except Exception:
        logger.warning(
            "codex fallback reroute failed", exc_info=True, extra={"reason": reason}
        )
        queue_failed(prompt, reason, path=queue_path)
        return LLMResult(text="", raw={"reason": reason})

    # Expose the routed model's text via ``result.text`` while preserving
    # provider specific metadata under ``result.raw``.
    if not getattr(result, "text", "").strip():
        logger.warning(
            "codex fallback produced no completion", extra={"reason": reason}
        )
        queue_failed(prompt, reason, path=queue_path)
        return LLMResult(text="", raw={"reason": reason})
    return result


__all__ = ["queue_failed", "reroute_to_fallback_model", "handle"]
