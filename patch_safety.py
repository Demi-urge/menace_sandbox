from __future__ import annotations

"""Assess patch safety via failure embeddings.

This module maintains a set of vector embeddings representing previously
observed failures.  Incoming patch metadata can be transformed into the same
vector space using :class:`error_vectorizer.ErrorVectorizer` and compared against
known failure vectors.  If a patch is too similar to past failures it can be
rejected or flagged before execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import math

from error_vectorizer import ErrorVectorizer


def _cosine(a: List[float], b: List[float]) -> float:
    """Return cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class PatchSafety:
    """Store failure embeddings and evaluate new patches."""

    threshold: float = 0.8
    vectorizer: ErrorVectorizer = field(default_factory=ErrorVectorizer)
    _failures: List[List[float]] = field(default_factory=list)

    def record_failure(self, err: Dict[str, Any]) -> None:
        """Add a failure example represented by ``err``."""
        # Ensure the vectoriser knows about categories/modules present in ``err``
        self.vectorizer.fit([err])
        self._failures.append(self.vectorizer.transform(err))

    def score(self, err: Dict[str, Any]) -> float:
        """Return the maximum similarity between ``err`` and recorded failures."""
        if not self._failures:
            return 0.0
        vec = self.vectorizer.transform(err)
        return max(_cosine(vec, f) for f in self._failures)

    def is_risky(self, err: Dict[str, Any]) -> bool:
        """Return ``True`` if ``err`` is similar to a stored failure."""
        return self.score(err) >= self.threshold


__all__ = ["PatchSafety"]
