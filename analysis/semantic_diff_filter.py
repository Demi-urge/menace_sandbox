from __future__ import annotations

"""Semantic diff filter using simple token embeddings.

This module embeds added code lines into lightweight token vectors and
compares them against a catalogue of unsafe pattern examples defined in
``unsafe_patterns``.  Examples include calls to ``eval``, ``subprocess`` with
``shell=True`` or higher level deserialisers like ``pickle.loads`` and
``yaml.load``.  The similarity metric is cosine similarity over the bag-of-words
representation which acts as a very small and fast embedding space.

The :func:`find_semantic_risks` helper is designed to be called with the
added lines from a diff hunk and returns any matches above the supplied
similarity threshold.
"""

import re
import math
from collections import Counter
from typing import Iterable, List, Tuple

from unsafe_patterns import UNSAFE_PATTERNS, UnsafePattern


def _embed(text: str) -> Counter[str]:
    """Return a bag-of-words vector for *text*."""
    tokens = re.findall(r"\w+", text.lower())
    return Counter(tokens)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between two token vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Pre-compute embeddings for all unsafe pattern examples.
_PATTERN_EMBEDS: List[Tuple[UnsafePattern, Counter[str]]] = [
    (pat, _embed(pat.example)) for pat in UNSAFE_PATTERNS
]


def find_semantic_risks(
    lines: Iterable[str], threshold: float = 0.5
) -> List[Tuple[str, str, float]]:
    """Return ``(line, pattern_message, score)`` for lines similar to patterns.

    ``threshold`` controls the minimum cosine similarity required for a line
    to be considered semantically similar to a known unsafe pattern.
    """

    matches: List[Tuple[str, str, float]] = []
    for line in lines:
        vec = _embed(line)
        if not vec:
            continue
        for pat, pvec in _PATTERN_EMBEDS:
            score = _cosine(vec, pvec)
            if score >= threshold:
                matches.append((line.strip(), pat.message, score))
    return matches


__all__ = ["find_semantic_risks"]
