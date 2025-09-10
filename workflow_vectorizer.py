from __future__ import annotations

"""Embedding based vectoriser for workflow specifications."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List
import json
import os
import re
import urllib.request
import numpy as np

from analysis.semantic_diff_filter import find_semantic_risks
from snippet_compressor import compress_snippets
from vector_utils import persist_embedding
from dynamic_path_router import resolve_path
try:  # pragma: no cover - event bus optional
    from unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - fallback
    UnifiedEventBus = None  # type: ignore

try:  # pragma: no cover - optional service
    from vector_service.vectorizer import SharedVectorService  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    SharedVectorService = None  # type: ignore

try:  # pragma: no cover - heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - fallback when package missing
    SentenceTransformer = None  # type: ignore

_MODEL = None
_EMBED_DIM = 384
if SentenceTransformer is not None:  # pragma: no cover - model download may be slow
    try:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBED_DIM = int(_MODEL.get_sentence_embedding_dimension())
    except Exception:
        _MODEL = None

_REMOTE_URL = os.environ.get("VECTOR_SERVICE_URL")


def _remote_embed(text: str) -> List[float]:
    data = json.dumps({"kind": "text", "record": {"text": text}}).encode("utf-8")
    req = urllib.request.Request(
        f"{_REMOTE_URL.rstrip('/')}/vectorise",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # pragma: no cover - network
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("vector", [])


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if _MODEL is not None:
        vecs = _MODEL.encode(texts)
        return [list(map(float, v)) for v in np.atleast_2d(vecs)]

    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is not None:
        try:
            return [_DEFAULT_SERVICE.vectorise("text", {"text": t}) for t in texts]
        except Exception:
            pass
    elif SharedVectorService is not None:
        try:
            _DEFAULT_SERVICE = SharedVectorService()
            return [_DEFAULT_SERVICE.vectorise("text", {"text": t}) for t in texts]
        except Exception:
            _DEFAULT_SERVICE = None

    if _REMOTE_URL:
        try:
            return [_remote_embed(t) for t in texts]
        except Exception:
            pass

    raise RuntimeError("No embedding backend available")


@dataclass
class WorkflowVectorizer:
    """Generate embeddings for workflow records."""

    _last_metrics: Dict[str, Any] = field(default_factory=dict, init=False)

    def fit(self, workflows: Iterable[Dict[str, Any]]) -> "WorkflowVectorizer":  # pragma: no cover - compatibility
        return self

    @property  # pragma: no cover - compatibility
    def dim(self) -> int:
        return _EMBED_DIM

    def graph_metrics(self) -> Dict[str, Any]:  # pragma: no cover - compatibility
        return self._last_metrics

    def transform(self, wf: Dict[str, Any], workflow_id: str | None = None) -> List[float]:
        parts: List[str] = []
        for key in ("name", "description", "category", "status"):
            val = wf.get(key)
            if isinstance(val, str):
                parts.append(val)

        steps = wf.get("workflow") or wf.get("task_sequence") or []
        if isinstance(steps, Iterable):
            for step in steps:
                if isinstance(step, dict):
                    for k in ("description", "function", "call", "name", "summary"):
                        sv = step.get(k)
                        if isinstance(sv, str):
                            parts.append(sv)
                else:
                    parts.append(str(step))

        text = "\n".join(parts)
        chunks: List[str] = []
        for sent in _split_sentences(text):
            if find_semantic_risks([sent]):
                continue
            summary = compress_snippets({"snippet": sent}).get("snippet", sent)
            if summary.strip():
                chunks.append(summary)

        if not chunks:
            vec = [0.0] * _EMBED_DIM
        else:
            vecs = _embed_texts(chunks)
            agg = np.mean(vecs, axis=0) if vecs else np.zeros(_EMBED_DIM)
            vec = [float(x) for x in agg.tolist()]
        self._last_metrics = {}
        return vec


try:
    _EMBEDDINGS_PATH = resolve_path("embeddings.jsonl").as_posix()
except Exception:  # pragma: no cover - defensive
    _EMBEDDINGS_PATH = (resolve_path(".") / "embeddings.jsonl").as_posix()

_DEFAULT_VECTORIZER = WorkflowVectorizer()
_DEFAULT_SERVICE = None
if SharedVectorService is not None:  # pragma: no cover - best effort
    try:
        _DEFAULT_SERVICE = SharedVectorService()
    except Exception:
        _DEFAULT_SERVICE = None

_EVENT_BUS = UnifiedEventBus() if UnifiedEventBus is not None else None


def vectorize_and_store(
    record_id: str,
    workflow: Dict[str, Any],
    *,
    path: str = _EMBEDDINGS_PATH,
    origin_db: str = "workflow",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``workflow`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(workflow, workflow_id=record_id)
    meta = metadata or {}
    if _DEFAULT_SERVICE is None:  # pragma: no cover - dependency unavailable
        raise RuntimeError("SharedVectorService unavailable")
    _DEFAULT_SERVICE.vectorise_and_store(
        "workflow",
        record_id,
        workflow,
        origin_db=origin_db,
        metadata=meta,
    )
    if _EVENT_BUS is not None:
        try:
            _EVENT_BUS.publish("db:record_updated", {"db": "workflow"})
        except Exception:
            pass
    return vec


def persist_workflow_embedding(
    record_id: str,
    workflow: Dict[str, Any],
    *,
    path: str = _EMBEDDINGS_PATH,
) -> List[float]:
    """Vectorise ``workflow`` and persist the embedding with metadata."""

    vec = _DEFAULT_VECTORIZER.transform(workflow, workflow_id=record_id)
    persist_embedding(
        "workflow",
        record_id,
        vec,
        origin_db="workflow",
        metadata=_DEFAULT_VECTORIZER.graph_metrics(),
        path=path,
    )
    if _EVENT_BUS is not None:
        try:
            _EVENT_BUS.publish("db:record_updated", {"db": "workflow"})
        except Exception:
            pass
    return vec


__all__ = ["WorkflowVectorizer", "vectorize_and_store", "persist_workflow_embedding"]
