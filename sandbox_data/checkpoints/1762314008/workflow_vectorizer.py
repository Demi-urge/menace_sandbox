from __future__ import annotations

"""Embedding based vectoriser for workflow specifications."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List
import re
import numpy as np

from analysis.semantic_diff_filter import find_semantic_risks
from snippet_compressor import compress_snippets
from vector_utils import persist_embedding
from dynamic_path_router import resolve_path
from vector_service.text_preprocessor import PreprocessingConfig, get_config, generalise
from vector_service.embed_utils import (
    get_text_embeddings,
    EMBED_DIM as _EMBED_DIM,
)
try:  # pragma: no cover - event bus optional
    from unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - fallback
    UnifiedEventBus = None  # type: ignore

try:  # pragma: no cover - optional service
    from vector_service.vectorizer import SharedVectorService  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    SharedVectorService = None  # type: ignore

_DEFAULT_SERVICE: SharedVectorService | None = None


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    return get_text_embeddings(texts, service=_DEFAULT_SERVICE)


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

    def transform(
        self,
        wf: Dict[str, Any],
        workflow_id: str | None = None,
        *,
        config: PreprocessingConfig | None = None,
    ) -> List[float]:
        cfg = config or get_config("workflow")
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
        sentences = _split_sentences(text) if cfg.split_sentences else [text]
        filtered: List[str] = []
        for sent in sentences:
            if cfg.filter_semantic_risks and find_semantic_risks([sent]):
                continue
            summary = compress_snippets({"snippet": sent}).get("snippet", sent)
            summary = generalise(summary, config=cfg, db_key="workflow")
            if summary.strip():
                filtered.append(summary)

        # chunk by approximate token counts
        chunks: List[str] = []
        if cfg.chunk_size and cfg.chunk_size > 0:
            current: List[str] = []
            count = 0
            for piece in filtered:
                count += len(piece.split())
                current.append(piece)
                if count >= cfg.chunk_size:
                    chunks.append(" ".join(current))
                    current = []
                    count = 0
            if current:
                chunks.append(" ".join(current))
        else:
            chunks = filtered

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
