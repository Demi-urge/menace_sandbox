from __future__ import annotations

"""Vectoriser for source code snippets using text embeddings."""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from chunking import split_into_chunks
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


@dataclass
class CodeVectorizer:
    """Generate embeddings for code content.

    The ``transform`` method extracts code, splits it into chunks, filters out
    risky lines and averages the remaining chunk embeddings.
    """

    max_tokens: int = 200

    def transform(
        self, rec: Dict[str, Any], *, config: PreprocessingConfig | None = None
    ) -> List[float]:
        code = str(rec.get("content") or "")
        if not code:
            return [0.0] * _EMBED_DIM

        cfg = config or get_config("code")
        token_limit = cfg.chunk_size or self.max_tokens

        chunks: List[str] = []
        for chunk in split_into_chunks(code, token_limit):
            if cfg.filter_semantic_risks and find_semantic_risks(chunk.text.splitlines()):
                continue
            summary = compress_snippets({"snippet": chunk.text}).get(
                "snippet", chunk.text
            )
            summary = generalise(summary, config=cfg, db_key="code")
            if summary.strip():
                chunks.append(summary)

        if not chunks:
            return [0.0] * _EMBED_DIM

        vecs = _embed_texts(chunks)
        agg = np.mean(vecs, axis=0) if vecs else np.zeros(_EMBED_DIM)
        return [float(x) for x in agg.tolist()]


__all__ = ["CodeVectorizer"]
