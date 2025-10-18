from __future__ import annotations

"""Shared vectorisation interface for disparate data sources.

This module consolidates the various standalone vectorisers into a single
service.  Callers provide a ``kind`` identifying the record type and a
dictionary representing the record.  The service delegates to the
appropriate vectoriser and optionally persists the resulting embedding
using a configurable :class:`~vector_service.vector_store.VectorStore`.
"""

# ruff: noqa: T201 - module level debug prints are routed via logging

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from pathlib import Path
import json
import logging
import os
import socket
import tarfile
import tempfile
import urllib.error
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


logger = logging.getLogger(__name__)


def _trace(event: str, **extra: Any) -> None:
    """Emit lightweight diagnostics when ``VECTOR_SERVICE_TRACE`` is set."""

    flag = os.getenv("VECTOR_SERVICE_TRACE", "")
    if flag and flag.lower() not in {"0", "false", "no", "off"}:
        payload = {"event": event, **extra}
        logger.log(logging.INFO, "vector-service: %s", event, extra=payload)


def _remote_timeout() -> float | None:
    raw = os.getenv("VECTOR_SERVICE_TIMEOUT", "10").strip()
    if not raw:
        return 10.0
    try:
        value = float(raw)
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_TIMEOUT=%r; defaulting to 10s", raw
        )
        return 10.0
    if value <= 0:
        logger.warning(
            "VECTOR_SERVICE_TIMEOUT must be positive; defaulting to 10s"
        )
        return 10.0
    return value


_REMOTE_URL = os.environ.get("VECTOR_SERVICE_URL")
_REMOTE_ENDPOINT = _REMOTE_URL.rstrip("/") if _REMOTE_URL else None
_REMOTE_TIMEOUT = _remote_timeout()
_REMOTE_DISABLED = False


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
        _trace("shared_vector_service.init.start")
        self._handlers = load_handlers()
        _trace(
            "shared_vector_service.handlers.loaded",
            handler_count=len(self._handlers),
            handlers=sorted(self._handlers.keys()),
        )
        if self.vector_store is None:
            _trace("shared_vector_service.vector_store.fetch")
            try:
                self.vector_store = get_default_vector_store()
            except Exception as exc:  # pragma: no cover - defensive logging
                _trace("shared_vector_service.vector_store.error", error=str(exc))
                raise
        _trace(
            "shared_vector_service.init.complete",
            has_vector_store=self.vector_store is not None,
        )

    def _encode_text(self, text: str) -> List[float]:
        if self.text_embedder is not None:
            _trace("shared_vector_service.encode_text.embedder", source="governed")
            vec = governed_embed(text, self.text_embedder)
            if vec is None:
                raise RuntimeError("embedding failed")
            return [float(x) for x in vec]
        if SentenceTransformer is None:
            # SentenceTransformer not installed: load bundled model
            _trace("shared_vector_service.encode_text.embedder", source="local_fallback")
            return _local_embed(text)
        raise RuntimeError("text embedder unavailable")

    def _call_remote(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        global _REMOTE_DISABLED
        if _REMOTE_DISABLED or _REMOTE_ENDPOINT is None:
            return None
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{_REMOTE_ENDPOINT}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                req, timeout=_REMOTE_TIMEOUT or None
            ) as resp:  # pragma: no cover - network
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            logger.warning(
                "remote vector service unavailable; falling back to local handling",  # pragma: no cover - diagnostics
                extra={"endpoint": _REMOTE_ENDPOINT, "path": path, "error": str(exc)},
            )
        except json.JSONDecodeError as exc:
            logger.warning(
                "remote vector service returned invalid JSON; falling back to local handling",  # pragma: no cover - diagnostics
                extra={"endpoint": _REMOTE_ENDPOINT, "path": path, "error": str(exc)},
            )
        _REMOTE_DISABLED = True
        return None

    def vectorise(self, kind: str, record: Dict[str, Any]) -> List[float]:
        """Return an embedding for ``record`` of type ``kind``."""
        _trace("shared_vector_service.vectorise.start", kind=kind)
        payload = self._call_remote("/vectorise", {"kind": kind, "record": record})
        if payload is not None:
            vec = payload.get("vector", [])
            if isinstance(vec, list):
                _trace("shared_vector_service.vectorise.remote", kind=kind)
                return vec

        kind = kind.lower()
        handler = self._handlers.get(kind)
        if handler:
            _trace("shared_vector_service.vectorise.handler", kind=kind)
            return handler(record)
        if kind in {"text", "prompt"}:
            _trace("shared_vector_service.vectorise.text", kind=kind)
            return self._encode_text(str(record.get("text", "")))
        if kind in {"stack"}:
            _trace("shared_vector_service.vectorise.stack", kind=kind)
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
        payload = self._call_remote(
            "/vectorise-and-store",
            {
                "kind": kind,
                "record_id": record_id,
                "record": record,
                "origin_db": origin_db,
                "metadata": metadata,
            },
        )
        if payload is not None:
            vec = payload.get("vector", vec)

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
