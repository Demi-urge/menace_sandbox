from __future__ import annotations

"""Embedding based vectoriser for bot definitions."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import re
import numpy as np

from analysis.semantic_diff_filter import find_semantic_risks
from snippet_compressor import compress_snippets
from vector_service.text_preprocessor import (
    PreprocessingConfig,
    get_config,
    generalise,
)
from vector_service.embed_utils import (
    get_text_embeddings as _embed_texts,
    EMBED_DIM as _EMBED_DIM,
)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@dataclass
class BotVectorizer:
    """Generate embeddings for bot configuration records."""

    def fit(self, bots: Iterable[Dict[str, Any]]) -> "BotVectorizer":  # pragma: no cover - compatibility
        return self

    @property  # pragma: no cover - compatibility
    def dim(self) -> int:
        return _EMBED_DIM

    def transform(
        self, bot: Dict[str, Any], *, config: PreprocessingConfig | None = None
    ) -> List[float]:
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

        cfg = config or get_config("bot")

        sentences = _split_sentences(text) if cfg.split_sentences else [text]
        filtered: List[str] = []
        for sent in sentences:
            if cfg.filter_semantic_risks and find_semantic_risks([sent]):
                continue
            summary = compress_snippets({"snippet": sent}).get("snippet", sent)
            summary = generalise(summary, config=cfg, db_key="bot")
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


__all__ = ["BotVectorizer"]
