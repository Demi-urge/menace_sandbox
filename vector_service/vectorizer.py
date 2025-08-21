from __future__ import annotations

"""Shared vectorisation interface for disparate data sources.

This module consolidates the various standalone vectorisers into a single
service.  Callers provide a ``kind`` identifying the record type and a
dictionary representing the record.  The service delegates to the
appropriate vectoriser and optionally persists the resulting embedding
using :mod:`vector_utils`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from action_vectorizer import ActionVectorizer
from error_vectorizer import ErrorVectorizer
from workflow_vectorizer import WorkflowVectorizer
from enhancement_vectorizer import EnhancementVectorizer
from bot_vectorizer import BotVectorizer
from vector_utils import persist_embedding
from governed_embeddings import governed_embed

try:  # pragma: no cover - optional dependency used for text embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - avoid hard dependency
    SentenceTransformer = None  # type: ignore


@dataclass
class SharedVectorService:
    """Facade exposing a unified ``vectorise`` API."""

    text_embedder: SentenceTransformer | None = None
    _action: ActionVectorizer = field(default_factory=ActionVectorizer)
    _error: ErrorVectorizer = field(default_factory=ErrorVectorizer)
    _workflow: WorkflowVectorizer = field(default_factory=WorkflowVectorizer)
    _enhancement: EnhancementVectorizer = field(default_factory=EnhancementVectorizer)
    _bot: BotVectorizer = field(default_factory=BotVectorizer)

    def _encode_text(self, text: str) -> List[float]:
        if self.text_embedder is None:
            raise RuntimeError("text embedder unavailable")
        vec = governed_embed(text, self.text_embedder)
        if vec is None:
            raise RuntimeError("embedding failed")
        return [float(x) for x in vec]

    def vectorise(self, kind: str, record: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``record`` of type ``kind``."""
        kind = kind.lower()
        if kind == "action":
            return self._action.transform(record)
        if kind == "error":
            return self._error.transform(record)
        if kind == "workflow":
            return self._workflow.transform(record)
        if kind == "enhancement":
            return self._enhancement.transform(record)
        if kind == "bot":
            return self._bot.transform(record)
        if kind in {"text", "prompt"}:
            return self._encode_text(str(record.get("text", "")))
        raise ValueError(f"unknown record type: {kind}")

    def vectorise_and_store(
        self, kind: str, record_id: str, record: Dict[str, Any]
    ) -> List[float]:
        """Vectorise ``record`` and persist the embedding.

        The embedding is written using :func:`vector_utils.persist_embedding`.
        """

        vec = self.vectorise(kind, record)
        if kind == "bot":
            persist_embedding("bot", record_id, vec)
        else:
            persist_embedding(kind, record_id, vec)
        return vec


__all__ = ["SharedVectorService"]
