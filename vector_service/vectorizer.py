from __future__ import annotations

"""Shared vectorisation interface for disparate data sources.

This module consolidates the various standalone vectorisers into a single
service.  Callers provide a ``kind`` identifying the record type and a
dictionary representing the record.  The service delegates to the
appropriate vectoriser and optionally persists the resulting embedding
using :mod:`vector_utils`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from pathlib import Path
import tarfile
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

from action_vectorizer import ActionVectorizer
from error_vectorizer import ErrorVectorizer
from workflow_vectorizer import WorkflowVectorizer
from enhancement_vectorizer import EnhancementVectorizer
from bot_vectorizer import BotVectorizer
from information_vectorizer import InformationVectorizer
from code_vectorizer import CodeVectorizer
from discrepancy_vectorizer import DiscrepancyVectorizer
from failure_vectorizer import FailureVectorizer
from research_vectorizer import ResearchVectorizer
from vector_utils import persist_embedding
from governed_embeddings import governed_embed
from .download_model import ensure_model

try:  # pragma: no cover - optional dependency used for text embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - avoid hard dependency
    SentenceTransformer = None  # type: ignore


_BUNDLED_MODEL = Path(__file__).with_name("minilm").joinpath(
    "tiny-distilroberta-base.tar.xz"
)
_LOCAL_TOKENIZER: AutoTokenizer | None = None
_LOCAL_MODEL: AutoModel | None = None


def _load_local_model() -> tuple[AutoTokenizer, AutoModel]:
    """Load the bundled fallback embedding model."""

    global _LOCAL_TOKENIZER, _LOCAL_MODEL
    if _LOCAL_TOKENIZER is None or _LOCAL_MODEL is None:
        ensure_model(_BUNDLED_MODEL)
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(_BUNDLED_MODEL) as tar:
                tar.extractall(tmpdir)
            _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(tmpdir)
            _LOCAL_MODEL = AutoModel.from_pretrained(tmpdir)
        _LOCAL_MODEL.eval()
    return _LOCAL_TOKENIZER, _LOCAL_MODEL


def _local_embed(text: str) -> List[float]:
    """Return an embedding using the bundled model."""

    tokenizer, model = _load_local_model()
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    embedding = output.last_hidden_state.mean(dim=1)[0]
    return [float(x) for x in embedding.tolist()]


@dataclass
class SharedVectorService:
    """Facade exposing a unified ``vectorise`` API."""

    text_embedder: SentenceTransformer | None = None
    _action: ActionVectorizer = field(default_factory=ActionVectorizer)
    _error: ErrorVectorizer = field(default_factory=ErrorVectorizer)
    _workflow: WorkflowVectorizer = field(default_factory=WorkflowVectorizer)
    _enhancement: EnhancementVectorizer = field(default_factory=EnhancementVectorizer)
    _bot: BotVectorizer = field(default_factory=BotVectorizer)
    _information: InformationVectorizer = field(default_factory=InformationVectorizer)
    _code: CodeVectorizer = field(default_factory=CodeVectorizer)
    _discrepancy: DiscrepancyVectorizer = field(default_factory=DiscrepancyVectorizer)
    _failure: FailureVectorizer = field(default_factory=FailureVectorizer)
    _research: ResearchVectorizer = field(default_factory=ResearchVectorizer)
    _handlers: Dict[str, Callable[[Dict[str, Any]], List[float]]] = field(init=False)

    def __post_init__(self) -> None:
        self._handlers = {
            "action": self._action.transform,
            "error": self._error.transform,
            "workflow": self._workflow.transform,
            "enhancement": self._enhancement.transform,
            "information": self._information.transform,
            "code": self._code.transform,
            "discrepancy": self._discrepancy.transform,
            "failure": self._failure.transform,
            "research": self._research.transform,
        }

    def _encode_text(self, text: str) -> List[float]:
        if self.text_embedder is not None:
            vec = governed_embed(text, self.text_embedder)
            if vec is None:
                raise RuntimeError("embedding failed")
            return [float(x) for x in vec]
        if SentenceTransformer is None:
            return _local_embed(text)
        raise RuntimeError("text embedder unavailable")

    def vectorise(self, kind: str, record: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``record`` of type ``kind``."""
        kind = kind.lower()
        if kind == "bot":
            return self._bot.transform(record)
        handler = self._handlers.get(kind)
        if handler:
            return handler(record)
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
        kind = kind.lower()
        if kind == "bot":
            persist_embedding("bot", record_id, vec)
        else:
            persist_embedding(kind, record_id, vec)
        return vec


__all__ = ["SharedVectorService"]
