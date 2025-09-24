from __future__ import annotations

"""Simple wrapper combining :mod:`gpt_memory` and :mod:`gpt_knowledge_service`.

This module exposes :class:`LocalKnowledgeModule` which couples
:class:`gpt_memory.GPTMemoryManager` with :class:`gpt_knowledge_service.GPTKnowledgeService`.
It provides a minimal interface used by bots that need persistent long-term
context.
"""

from pathlib import Path
from typing import Sequence, Any

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - keep import lightweight
    SentenceTransformer = None  # type: ignore

from governed_embeddings import get_embedder

try:
    from .gpt_memory import GPTMemoryManager
except ImportError:  # pragma: no cover - allow flat imports
    try:
        from menace_sandbox.gpt_memory import GPTMemoryManager  # type: ignore
    except ImportError:  # pragma: no cover - flat layout fallback
        from gpt_memory import GPTMemoryManager  # type: ignore
from gpt_knowledge_service import GPTKnowledgeService
from vector_service import CognitionLayer
try:  # pragma: no cover - allow flat imports
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX  # type: ignore


class LocalKnowledgeModule:
    """Aggregate GPT memory and summarised insights.

    Parameters
    ----------
    db_path:
        Location of the SQLite database used for storing interactions.  If an
        existing :class:`GPTMemoryManager` is supplied it will be used instead.
    manager:
        Optional pre-initialised :class:`GPTMemoryManager` instance.
    service:
        Optional pre-initialised :class:`GPTKnowledgeService` instance.  When not
        provided one will be created using ``manager``.
    embedder:
        Optional :class:`SentenceTransformer` shared with the underlying
        :class:`GPTMemoryManager` to enable semantic search.
    """

    def __init__(
        self,
        db_path: str | Path = "gpt_memory.db",
        *,
        manager: GPTMemoryManager | None = None,
        service: GPTKnowledgeService | None = None,
        embedder: "SentenceTransformer | None" = None,
        cognition_layer: CognitionLayer | None = None,
    ) -> None:
        self.memory = manager or GPTMemoryManager(db_path, embedder=embedder)
        self.knowledge = service or GPTKnowledgeService(self.memory)
        self.cognition_layer = cognition_layer

    # ------------------------------------------------------------------ facade
    def log(
        self, prompt: str, response: str, tags: Sequence[str] | None = None
    ) -> None:
        """Store an interaction in long-term memory."""

        self.memory.log_interaction(prompt, response, tags)

    def get_insights(self, tag: str) -> str:
        """Return the latest generated insight for ``tag``."""

        return self.knowledge.get_recent_insights(tag)

    def refresh(self) -> None:
        """Regenerate stored insights from recent interactions."""

        self.knowledge.update_insights()

    # ---------------------------------------------------------------- context
    def _fmt(self, entries: Sequence[Any], title: str) -> str:
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

    def build_context(self, key: str, limit: int = 5) -> str:
        """Return formatted context for ``key`` using :class:`CognitionLayer`.

        Falls back to the legacy memory aggregation when no
        :class:`CognitionLayer` instance is available.
        """

        if self.cognition_layer is not None:
            try:
                ctx, _ = self.cognition_layer.query(key, top_k=limit)
                return ctx
            except Exception:
                return ""

        sections: list[str] = []
        try:
            fb = self.memory.search_context(
                key, tags=[FEEDBACK], limit=limit, use_embeddings=True
            )
            ctx = self._fmt(fb, "Feedback")
            if ctx:
                sections.append(ctx)
        except Exception:
            pass
        try:
            fixes = self.memory.search_context(
                key, tags=[ERROR_FIX], limit=limit, use_embeddings=True
            )
            ctx = self._fmt(fixes, "Error fixes")
            if ctx:
                sections.append(ctx)
        except Exception:
            pass
        try:
            improv = self.memory.search_context(
                key, tags=[IMPROVEMENT_PATH], limit=limit, use_embeddings=True
            )
            ctx = self._fmt(improv, "Improvement paths")
            if ctx:
                sections.append(ctx)
        except Exception:
            pass

        try:
            insight = self.knowledge.get_recent_insights(FEEDBACK)
            if insight:
                sections.append(f"### Feedback insight\n- {insight}")
        except Exception:
            pass
        try:
            insight = self.knowledge.get_recent_insights(ERROR_FIX)
            if insight:
                sections.append(f"### Error fix insight\n- {insight}")
        except Exception:
            pass
        try:
            insight = self.knowledge.get_recent_insights(IMPROVEMENT_PATH)
            if insight:
                sections.append(f"### Improvement path insight\n- {insight}")
        except Exception:
            pass

        return "\n\n".join(sections)


_LOCAL_KNOWLEDGE: "LocalKnowledgeModule | None" = None


def init_local_knowledge(mem_db: str | Path) -> LocalKnowledgeModule:
    """Return a process-wide :class:`LocalKnowledgeModule` instance.

    The module is initialised on first use and subsequent calls return the
    existing instance, ensuring that all components share the same underlying
    :class:`GPTMemoryManager` and :class:`GPTKnowledgeService`.
    """

    global _LOCAL_KNOWLEDGE
    if _LOCAL_KNOWLEDGE is None:
        _LOCAL_KNOWLEDGE = LocalKnowledgeModule(mem_db, embedder=get_embedder())
    return _LOCAL_KNOWLEDGE


__all__ = ["LocalKnowledgeModule", "init_local_knowledge"]
