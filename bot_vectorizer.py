from __future__ import annotations

"""Embedding based vectoriser for bot definitions."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import json
import os
import re
import urllib.request
import numpy as np

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
class BotVectorizer:
    """Generate embeddings for bot configuration records."""

    def fit(self, bots: Iterable[Dict[str, Any]]) -> "BotVectorizer":  # pragma: no cover - compatibility
        return self

    @property  # pragma: no cover - compatibility
    def dim(self) -> int:
        return _EMBED_DIM

    def transform(self, bot: Dict[str, Any]) -> List[float]:
        parts: List[str] = []
        for key in ("name", "description", "summary", "goal", "status", "type"):
            val = bot.get(key)
            if isinstance(val, str):
                parts.append(val)
        tasks = bot.get("tasks") or []
        if isinstance(tasks, Iterable) and not isinstance(tasks, str):
            parts.extend(str(t) for t in tasks)
        elif isinstance(tasks, str):
            parts.append(tasks)

        text = "\n".join(parts)
        if not text:
            return [0.0] * _EMBED_DIM

        chunks: List[str] = []
        for sent in _split_sentences(text):
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


__all__ = ["BotVectorizer"]
