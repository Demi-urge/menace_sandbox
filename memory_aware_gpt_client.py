from __future__ import annotations

"""Wrapper for ChatGPTClient with memory-based context injection.

Context is sourced from a :class:`local_knowledge_module.LocalKnowledgeModule`
which provides access to both raw memory entries and summarised insights.
The collected context is prepended to the prompt before delegating to
``ChatGPTClient`` and the interaction is logged back to the module.
"""

from typing import Sequence, Any

try:  # pragma: no cover - allow flat imports
    from .local_knowledge_module import LocalKnowledgeModule
except Exception:  # pragma: no cover - fallback for flat layout
    from local_knowledge_module import LocalKnowledgeModule  # type: ignore


__all__ = ["ask_with_memory"]


def ask_with_memory(
    client: Any,
    key: str,
    prompt: str,
    *,
    memory: LocalKnowledgeModule,
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
        :class:`LocalKnowledgeModule` used to build context and record
        interactions.
    tags:
        Tags applied when logging the interaction.
    """

    try:
        ctx = memory.build_context(key, limit=5)
    except Exception:
        ctx = ""
    full_prompt = f"{ctx}\n\n{prompt}" if ctx else prompt

    messages = [{"role": "user", "content": full_prompt}]
    data = client.ask(messages, use_memory=False, memory_manager=None, tags=tags)
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    try:
        memory.log(full_prompt, text, tags)
    except Exception:
        pass
    return data
