from __future__ import annotations

"""Wrapper for ChatGPTClient with memory and vector-based context injection.

Context is sourced from a :class:`local_knowledge_module.LocalKnowledgeModule`
which provides access to both raw memory entries and summarised insights.
Additional contextual snippets are retrieved from the vector service via a
``ContextBuilder``.  The collected context is prepended to the prompt before
delegating to ``ChatGPTClient`` and the interaction is logged back to the
module.
"""

from typing import Sequence, Any, Dict
import logging
import uuid

try:  # pragma: no cover - optional dependency
    from memory_logging import ensure_tags
except Exception:  # pragma: no cover - fallback when logging unavailable
    def ensure_tags(key: str, tags: Sequence[str] | None) -> list[str]:
        return [key, *(tags or [])]

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback when vector service missing
    ContextBuilder = Any  # type: ignore

try:  # pragma: no cover - optional dependency
    from prompt_types import Prompt
except Exception:  # pragma: no cover - minimal stub
    class Prompt:  # type: ignore
        def __init__(self, user: str = "", **_: Any) -> None:
            self.system = ""
            self.user = user
            self.examples: list[str] = []
            self.metadata: Dict[str, Any] = {}

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


logger = logging.getLogger(__name__)
def ask_with_memory(
    client: Any,
    key: str,
    prompt: str | Prompt,
    *,
    memory: LocalKnowledgeModule,
    context_builder: ContextBuilder,
    tags: Sequence[str] | None = None,
    intent: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> str:
    """Query ``client`` with ``prompt`` augmented by prior context.

    ``prompt`` may be either a raw string or a pre-built :class:`Prompt`.
    When a :class:`Prompt` is provided, it is used directly and any supplied
    ``intent`` metadata is merged into its ``metadata`` field.

    Parameters
    ----------
    client:
        Object implementing an ``ask`` method compatible with
        :class:`ChatGPTClient`.
    key:
        Identifier used to retrieve related feedback, improvement paths and
        error fixes from ``memory``.
    prompt:
        The new user prompt or existing :class:`Prompt`.
    memory:
        :class:`LocalKnowledgeModule` used to build context and record
        interactions.
    context_builder:
        :class:`ContextBuilder` used to retrieve vector-based context snippets
        when ``prompt`` is a raw string.
    tags:
        Tags applied when logging the interaction.
    intent:
        Optional intent metadata forwarded to ``context_builder`` or merged into
        ``prompt`` when provided.
    metadata:
        Backwards compatible alias for ``intent``.
    """

    intent_payload = intent or metadata

    try:
        mem_ctx = memory.build_context(key, limit=5)
    except Exception:
        mem_ctx = ""

    session_id = uuid.uuid4().hex
    if isinstance(prompt, Prompt):
        base_system = prompt.system
        extra_examples = list(getattr(prompt, "examples", []))
        if intent_payload:
            try:
                prompt.metadata.update(intent_payload)
            except Exception:
                prompt.metadata = dict(intent_payload)
        orig_meta = getattr(prompt, "metadata", {})
        prompt_text = prompt.user
    else:
        base_system = ""
        extra_examples = []
        orig_meta = {}
        prompt_text = prompt

    intent_meta: Dict[str, Any] = dict(intent_payload or {})
    if mem_ctx:
        intent_meta["memory_context"] = mem_ctx
    intent_meta.setdefault("user_query", prompt_text)

    try:
        prompt_obj = context_builder.build_prompt(
            prompt_text, intent_metadata=intent_meta, session_id=session_id
        )
    except Exception:
        logger.exception("ContextBuilder.build_prompt failed")
        raise

    if extra_examples or mem_ctx:
        merged = list(extra_examples)
        if mem_ctx:
            merged.append(mem_ctx)
        merged.extend(getattr(prompt_obj, "examples", []))
        prompt_obj.examples = merged
    if base_system:
        prompt_obj.system = base_system

    full_tags = ensure_tags(key, tags)
    meta = getattr(prompt_obj, "metadata", {}) or {}
    meta.update(orig_meta)
    meta.setdefault("session_id", session_id)
    meta.setdefault("tags", full_tags)
    if mem_ctx:
        meta.setdefault("memory_context", mem_ctx)
    retrieved_context = "\n".join(getattr(prompt_obj, "examples", []))
    if retrieved_context:
        meta.setdefault("retrieved_context", retrieved_context)
    prompt_obj.metadata = meta

    data = client.ask(
        prompt_obj, use_memory=False, memory_manager=None, tags=full_tags
    )
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    try:
        log_prompt = "\n\n".join(prompt_obj.examples + [prompt_obj.user])
        if prompt_obj.system:
            log_prompt = f"{prompt_obj.system}\n\n{log_prompt}"
        memory.log(log_prompt, text, full_tags)
    except Exception:
        pass
    return text
