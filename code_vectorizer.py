from __future__ import annotations

"""Vectoriser for source code snippets using text embeddings."""

from dataclasses import dataclass
from typing import Any, Dict, List
import json
import os
import urllib.request
import numpy as np

from chunking import split_into_chunks
from analysis.semantic_diff_filter import find_semantic_risks
from snippet_compressor import compress_snippets

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
    try:  # pragma: no cover - defensive
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
class CodeVectorizer:
    """Generate embeddings for code content.

    The ``transform`` method extracts code, splits it into chunks, filters out
    risky lines and averages the remaining chunk embeddings.
    """

    max_tokens: int = 200

    def transform(self, rec: Dict[str, Any]) -> List[float]:
        code = str(rec.get("content") or "")
        if not code:
            return [0.0] * _EMBED_DIM

        chunks: List[str] = []
        for chunk in split_into_chunks(code, self.max_tokens):
            if find_semantic_risks(chunk.text.splitlines()):
                continue
            summary = compress_snippets({"snippet": chunk.text}).get(
                "snippet", chunk.text
            )
            if summary.strip():
                chunks.append(summary)

        if not chunks:
            return [0.0] * _EMBED_DIM

        vecs = _embed_texts(chunks)
        agg = np.mean(vecs, axis=0) if vecs else np.zeros(_EMBED_DIM)
        return [float(x) for x in agg.tolist()]


__all__ = ["CodeVectorizer"]
