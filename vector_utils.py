"""Utility functions for vector operations and persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Return cosine similarity between vectors ``a`` and ``b``."""
    vec_a = list(a)
    vec_b = list(b)
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def persist_embedding(
    kind: str,
    record_id: str,
    embedding: Sequence[float],
    *,
    path: str | Path = "embeddings.jsonl",
) -> None:
    """Append ``embedding`` to ``path`` with minimal metadata.

    The data is stored in JSON lines format making it easy to stream into
    other systems.  Existing files are appended to and created on demand.
    """

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": kind,
        "id": record_id,
        "vector": [float(x) for x in embedding],
    }
    with file_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
