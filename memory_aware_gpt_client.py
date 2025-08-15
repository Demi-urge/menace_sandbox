from __future__ import annotations

"""Wrapper for ChatGPTClient with memory-based context injection.

This helper retrieves prior feedback, improvement paths and error fixes for a
module/action key using :mod:`knowledge_retriever`.  The collected context is
prepended to the prompt before delegating to ``ChatGPTClient``.  Each
interaction is logged via :func:`memory_logging.log_with_tags`.
"""

from typing import Sequence, Any

try:  # pragma: no cover - allow flat imports
    from .knowledge_retriever import (
        get_feedback,
        get_improvement_paths,
        get_error_fixes,
    )
except Exception:  # pragma: no cover - fallback for flat layout
    from knowledge_retriever import (  # type: ignore
        get_feedback,
        get_improvement_paths,
        get_error_fixes,
    )

try:  # pragma: no cover - allow flat imports
    from .memory_logging import log_with_tags
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_logging import log_with_tags  # type: ignore


__all__ = ["ask_with_memory"]


def _fmt(entries: Sequence[Any], title: str) -> str:
    parts: list[str] = []
    for e in entries:
        prompt = getattr(e, "prompt", "")
        response = getattr(e, "response", "")
        text = response or prompt
        if text:
            parts.append(f"- {text}")
    if not parts:
        return ""
    body = "\n".join(parts)
    return f"### {title}\n{body}"


def ask_with_memory(
    client: Any,
    key: str,
    prompt: str,
    *,
    memory: Any,
    tags: Sequence[str] | None = None,
) -> dict:
    """Query ``client`` with ``prompt`` augmented by prior context.

    Parameters
    ----------
    client:
        Object implementing an ``ask`` method compatible with
        :class:`ChatGPTClient`.
    key:
        Identifier used to retrieve related feedback, improvement paths and
        error fixes from ``memory``.
    prompt:
        The new user prompt.
    memory:
        GPT memory instance passed to :func:`knowledge_retriever` and used for
        logging.
    tags:
        Tags applied when logging the interaction.
    """

    ctx_parts: list[str] = []
    try:
        fb = get_feedback(memory, key, limit=5)
        ctx = _fmt(fb, "Feedback")
        if ctx:
            ctx_parts.append(ctx)
    except Exception:
        pass
    try:
        fixes = get_error_fixes(memory, key, limit=3)
        ctx = _fmt(fixes, "Error fixes")
        if ctx:
            ctx_parts.append(ctx)
    except Exception:
        pass
    try:
        improv = get_improvement_paths(memory, key, limit=3)
        ctx = _fmt(improv, "Improvement paths")
        if ctx:
            ctx_parts.append(ctx)
    except Exception:
        pass

    full_prompt = prompt
    if ctx_parts:
        full_prompt = "\n\n".join(ctx_parts) + "\n\n" + prompt

    messages = [{"role": "user", "content": full_prompt}]
    data = client.ask(messages, use_memory=False, memory_manager=None, tags=tags)
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    try:
        log_with_tags(memory, full_prompt, text, tags)
    except Exception:
        pass
    return data
