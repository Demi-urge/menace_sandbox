from __future__ import annotations

"""Shared vectorisation interface for disparate data sources.

This module consolidates the various standalone vectorisers into a single
service.  Callers provide a ``kind`` identifying the record type and a
dictionary representing the record.  The service delegates to the
appropriate vectoriser and optionally persists the resulting embedding
using a configurable :class:`~vector_service.vector_store.VectorStore`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from pathlib import Path
import os
import tarfile
import tempfile
import json
import urllib.request

import torch
from transformers import AutoModel, AutoTokenizer

from dynamic_path_router import resolve_path

from governed_embeddings import governed_embed
from .registry import load_handlers
from .vector_store import VectorStore, get_default_vector_store

try:  # pragma: no cover - optional dependency used for text embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - avoid hard dependency
    SentenceTransformer = None  # type: ignore


_BUNDLED_MODEL = resolve_path("vector_service/minilm") / "tiny-distilroberta-base.tar.xz"
_LOCAL_TOKENIZER: AutoTokenizer | None = None
_LOCAL_MODEL: AutoModel | None = None


_REMOTE_URL = os.environ.get("VECTOR_SERVICE_URL")


def _load_local_model() -> tuple[AutoTokenizer, AutoModel]:
    """Load the bundled fallback embedding model."""

    global _LOCAL_TOKENIZER, _LOCAL_MODEL
    if _LOCAL_TOKENIZER is None or _LOCAL_MODEL is None:
        if not _BUNDLED_MODEL.exists():
            raise FileNotFoundError(
                f"bundled model archive missing at {_BUNDLED_MODEL} "
                "- run `python -m vector_service.download_model` to fetch it"
            )
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
    vector_store: VectorStore | None = None
    _handlers: Dict[str, Callable[[Dict[str, Any]], List[float]]] = field(init=False)

    def __post_init__(self) -> None:
        # Handlers are populated dynamically from the registry so newly
        # registered vectorisers are picked up automatically.
        self._handlers = load_handlers()
        if self.vector_store is None:
            self.vector_store = get_default_vector_store()

    def _encode_text(self, text: str) -> List[float]:
        if self.text_embedder is not None:
            vec = governed_embed(text, self.text_embedder)
            if vec is None:
                raise RuntimeError("embedding failed")
            return [float(x) for x in vec]
        if SentenceTransformer is None:
            # SentenceTransformer not installed: load bundled model
            return _local_embed(text)
        raise RuntimeError("text embedder unavailable")

    def vectorise(self, kind: str, record: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``record`` of type ``kind``."""
        if _REMOTE_URL:
            data = json.dumps({"kind": kind, "record": record}).encode("utf-8")
            req = urllib.request.Request(
                f"{_REMOTE_URL.rstrip('/')}/vectorise",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:  # pragma: no cover - network
                payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("vector", [])

        kind = kind.lower()
        handler = self._handlers.get(kind)
        if handler:
            return handler(record)
        if kind in {"text", "prompt"}:
            return self._encode_text(str(record.get("text", "")))
        raise ValueError(f"unknown record type: {kind}")

    def vectorise_and_store(
        self,
        kind: str,
        record_id: str,
        record: Dict[str, Any],
        *,
        origin_db: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> List[float]:
        """Vectorise ``record`` and persist the embedding."""

        vec = self.vectorise(kind, record)
        if _REMOTE_URL:
            data = json.dumps(
                {
                    "kind": kind,
                    "record_id": record_id,
                    "record": record,
                    "origin_db": origin_db,
                    "metadata": metadata,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{_REMOTE_URL.rstrip('/')}/vectorise-and-store",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:  # pragma: no cover - network
                payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("vector", [])

        if self.vector_store is None:
            raise RuntimeError("VectorStore not configured")
        self.vector_store.add(
            kind.lower(),
            record_id,
            vec,
            origin_db=origin_db or kind,
            metadata=metadata or {},
        )
        return vec


def update_workflow_embeddings(db_path: str = "workflows.db") -> None:
    """Embed all workflows in ``db_path`` using :class:`SharedVectorService`."""

    try:  # pragma: no cover - optional dependency
        from dataclasses import asdict
        from task_handoff_bot import WorkflowDB  # type: ignore
    except Exception:  # pragma: no cover - best effort
        return

    svc = SharedVectorService()
    db = WorkflowDB(Path(db_path))
    for wid, rec, _ in db.iter_records():
        try:
            svc.vectorise_and_store(
                "workflow",
                str(wid),
                asdict(rec),
                origin_db="workflow",
            )
        except Exception:  # pragma: no cover - best effort
            continue

__all__ = ["SharedVectorService", "update_workflow_embeddings"]
