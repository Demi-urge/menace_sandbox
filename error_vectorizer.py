from __future__ import annotations

"""Embedding based vectoriser for error telemetry records."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import re
import numpy as np

from analysis.semantic_diff_filter import find_semantic_risks
from snippet_compressor import compress_snippets
from chunking import split_into_chunks
from vector_utils import persist_embedding

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


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    if _MODEL is None:  # pragma: no cover - dependency missing
        return [[0.0] * _EMBED_DIM for _ in texts]
    vecs = _MODEL.encode(texts)
    return [list(map(float, v)) for v in np.atleast_2d(vecs)]


@dataclass
class ErrorVectorizer:
    """Generate embeddings for error log entries."""

    max_tokens: int = 200

    def fit(self, errors: Iterable[Dict[str, Any]]) -> "ErrorVectorizer":  # pragma: no cover - compatibility
        return self

    @property  # pragma: no cover - compatibility
    def dim(self) -> int:
        return _EMBED_DIM

    def transform(self, err: Dict[str, Any]) -> List[float]:
        chunks: List[str] = []

        msg = err.get("message") or err.get("error") or ""
        if isinstance(msg, str):
            for sent in _split_sentences(msg):
                if find_semantic_risks([sent]):
                    continue
                summary = compress_snippets({"snippet": sent}).get("snippet", sent)
                if summary.strip():
                    chunks.append(summary)

        stack = err.get("stack_trace") or err.get("stack") or ""
        if isinstance(stack, str) and stack.strip():
            for ch in split_into_chunks(stack, self.max_tokens):
                if find_semantic_risks(ch.text.splitlines()):
                    continue
                summary = compress_snippets({"snippet": ch.text}).get(
                    "snippet", ch.text
                )
                if summary.strip():
                    chunks.append(summary)

        other = [err.get("category"), err.get("module"), err.get("root_module")]
        for item in other:
            if isinstance(item, str) and item.strip():
                sent = item.strip()
                if find_semantic_risks([sent]):
                    continue
                summary = compress_snippets({"snippet": sent}).get("snippet", sent)
                if summary.strip():
                    chunks.append(summary)

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
