from __future__ import annotations

"""Wrapper for ChatGPTClient with memory and vector-based context injection.

Context is sourced from a :class:`local_knowledge_module.LocalKnowledgeModule`
which provides access to both raw memory entries and summarised insights.
Additional contextual snippets are retrieved from the vector service via a
``ContextBuilder``.  The collected context is prepended to the prompt before
delegating to ``ChatGPTClient`` and the interaction is logged back to the
module.
"""

from typing import Sequence, Any
import uuid

try:  # pragma: no cover - optional dependency
    from memory_logging import ensure_tags
except Exception:  # pragma: no cover - fallback when logging unavailable
    def ensure_tags(key: str, tags: Sequence[str] | None) -> list[str]:
        return [key, *(tags or [])]
from snippet_compressor import compress_snippets

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder, FallbackResult
except Exception:  # pragma: no cover - fallback when vector service missing
    ContextBuilder = Any  # type: ignore

    class FallbackResult(list):  # type: ignore
        """Fallback placeholder when vector service is unavailable."""

try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback when service missing
    class ErrorResult(Exception):  # type: ignore
        """Fallback placeholder when vector service is unavailable."""

try:  # pragma: no cover - allow flat imports
    from .local_knowledge_module import LocalKnowledgeModule
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from local_knowledge_module import LocalKnowledgeModule  # type: ignore
    except Exception:  # pragma: no cover - minimal stub when module unavailable
        class LocalKnowledgeModule:  # type: ignore
            def build_context(self, key: str, limit: int = 5) -> str:  # noqa: D401
                return ""

            def log(self, prompt: str, resp: str, tags) -> None:  # noqa: D401
                pass


__all__ = ["ask_with_memory"]


def ask_with_memory(
    client: Any,
    key: str,
    prompt: str,
    *,
    memory: LocalKnowledgeModule,
    context_builder: ContextBuilder,
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
    context_builder:
        :class:`ContextBuilder` used to retrieve vector-based context snippets.
    tags:
        Tags applied when logging the interaction.
    """

    try:
        ctx = memory.build_context(key, limit=5)
    except Exception:
        ctx = ""
    full_prompt = f"{ctx}\n\n{prompt}" if ctx else prompt

    vec_ctx = ""
    session_id = uuid.uuid4().hex
    try:
        ctx_res = context_builder.build(key, session_id=session_id)
        vec_ctx = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
        if isinstance(vec_ctx, (FallbackResult, ErrorResult)):
            vec_ctx = ""
        elif vec_ctx:
            vec_ctx = compress_snippets({"snippet": vec_ctx}).get(
                "snippet", vec_ctx
            )
    except Exception:
        vec_ctx = ""
    if vec_ctx:
        full_prompt = f"{vec_ctx}\n\n{full_prompt}"

    messages = [{"role": "user", "content": full_prompt}]
    full_tags = ensure_tags(key, tags)
    data = client.ask(
        messages, use_memory=False, memory_manager=None, tags=full_tags
    )
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    try:
        memory.log(full_prompt, text, full_tags)
    except Exception:
        pass
    return data
