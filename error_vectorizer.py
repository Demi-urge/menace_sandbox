from __future__ import annotations

"""Embedding based vectoriser for error telemetry records."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import json
import os
import re
import urllib.request
import numpy as np

from analysis.semantic_diff_filter import find_semantic_risks
from snippet_compressor import compress_snippets
from chunking import split_into_chunks
from vector_utils import persist_embedding
from vector_service.text_preprocessor import (
    PreprocessingConfig,
    get_config,
    generalise,
)

try:  # pragma: no cover - heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - fallback when package missing
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional service
    from vector_service.vectorizer import SharedVectorService  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    SharedVectorService = None  # type: ignore

_MODEL = None
_EMBED_DIM = 384
if SentenceTransformer is not None:  # pragma: no cover - model download may be slow
    try:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBED_DIM = int(_MODEL.get_sentence_embedding_dimension())
    except Exception:
        _MODEL = None

_SERVICE: SharedVectorService | None = None
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

    global _SERVICE
    if SharedVectorService is not None:
        if _SERVICE is None:
            try:
                embedder = None
                if SentenceTransformer is not None:
                    try:
                        embedder = SentenceTransformer("all-MiniLM-L6-v2")
                    except Exception:
                        embedder = None
                _SERVICE = SharedVectorService(embedder)
            except Exception:
                _SERVICE = None
        if _SERVICE is not None:
            try:
                return [_SERVICE.vectorise("text", {"text": t}) for t in texts]
            except Exception:
                pass

    if _REMOTE_URL:
        try:
            return [_remote_embed(t) for t in texts]
        except Exception:
            pass

    raise RuntimeError("No embedding backend available")


@dataclass
class ErrorVectorizer:
    """Generate embeddings for error log entries."""

    max_tokens: int = 200

    def fit(self, errors: Iterable[Dict[str, Any]]) -> "ErrorVectorizer":  # pragma: no cover - compatibility
        return self

    @property  # pragma: no cover - compatibility
    def dim(self) -> int:
        return _EMBED_DIM

    def transform(
        self, err: Dict[str, Any], *, config: PreprocessingConfig | None = None
    ) -> List[float]:
        cfg = config or get_config("error")
        filtered: List[str] = []

        msg = err.get("message") or err.get("error") or ""
        if isinstance(msg, str):
            sentences = _split_sentences(msg) if cfg.split_sentences else [msg]
            for sent in sentences:
                if cfg.filter_semantic_risks and find_semantic_risks([sent]):
                    continue
                summary = compress_snippets({"snippet": sent}).get("snippet", sent)
                summary = generalise(summary, config=cfg, db_key="error")
                if summary.strip():
                    filtered.append(summary)

        stack = err.get("stack_trace") or err.get("stack") or ""
        if isinstance(stack, str) and stack.strip():
            for ch in split_into_chunks(stack, cfg.chunk_size or self.max_tokens):
                if cfg.filter_semantic_risks and find_semantic_risks(ch.text.splitlines()):
                    continue
                summary = compress_snippets({"snippet": ch.text}).get(
                    "snippet", ch.text
                )
                summary = generalise(summary, config=cfg, db_key="error")
                if summary.strip():
                    filtered.append(summary)

        other = [err.get("category"), err.get("module"), err.get("root_module")]
        for item in other:
            if isinstance(item, str) and item.strip():
                sent = item.strip()
                if cfg.filter_semantic_risks and find_semantic_risks([sent]):
                    continue
                summary = compress_snippets({"snippet": sent}).get("snippet", sent)
                summary = generalise(summary, config=cfg, db_key="error")
                if summary.strip():
                    filtered.append(summary)

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
            return [0.0] * _EMBED_DIM

        vecs = _embed_texts(chunks)
        agg = np.mean(vecs, axis=0) if vecs else np.zeros(_EMBED_DIM)
        return [float(x) for x in agg.tolist()]


_DEFAULT_VECTORIZER = ErrorVectorizer()


def vectorize_and_store(
    record_id: str,
    error: Dict[str, Any],
    *,
    path: str = "embeddings.jsonl",
    origin_db: str = "error",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``error`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(error)
    try:
        persist_embedding(
            "error",
            record_id,
            vec,
            path=path,
            origin_db=origin_db,
            metadata=metadata or {},
        )
    except TypeError:  # pragma: no cover - compatibility with older signatures
        persist_embedding("error", record_id, vec, path=path)
    return vec


__all__ = ["ErrorVectorizer", "vectorize_and_store"]
