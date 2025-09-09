from __future__ import annotations

"""Embedding based vectoriser for bot definitions."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import re
import numpy as np

from analysis.semantic_diff_filter import find_semantic_risks
from snippet_compressor import compress_snippets

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
