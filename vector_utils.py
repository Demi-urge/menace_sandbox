"""Utility functions for vector operations."""

from __future__ import annotations

import math
from typing import Iterable


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Return cosine similarity between vectors ``a`` and ``b``."""
    vec_a = list(a)
    vec_b = list(b)
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
