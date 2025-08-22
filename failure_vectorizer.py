from __future__ import annotations

"""Vectoriser for failure learning system records."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

_DEFAULT_BOUNDS = {
    "profitability": 1_000_000.0,
    "retention": 100.0,
    "cac": 100_000.0,
    "roi": 1000.0,
}

def _one_hot(idx: int, length: int) -> List[float]:
    vec = [0.0] * length
    if 0 <= idx < length:
        vec[idx] = 1.0
    return vec

def _get_index(value: Any, mapping: Dict[str, int], max_size: int) -> int:
    val = str(value).lower().strip() or "other"
    if val in mapping:
        return mapping[val]
    if len(mapping) < max_size:
        mapping[val] = len(mapping)
        return mapping[val]
    return mapping["other"]

def _scale(value: Any, bound: float) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    f = max(-bound, min(bound, f))
    return f / bound if bound else 0.0

@dataclass
class FailureVectorizer:
    """Lightweight vectoriser for ``failures.db`` records."""

    max_causes: int = 20
    max_demographics: int = 20
    cause_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    demo_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})

    def fit(self, records: Iterable[Dict[str, Any]]) -> "FailureVectorizer":
        for rec in records:
            _get_index(rec.get("cause"), self.cause_index, self.max_causes)
            _get_index(rec.get("demographics"), self.demo_index, self.max_demographics)
        return self

    @property
    def dim(self) -> int:
        return self.max_causes + self.max_demographics + 4

    def transform(self, rec: Dict[str, Any]) -> List[float]:
        c_idx = _get_index(rec.get("cause"), self.cause_index, self.max_causes)
        d_idx = _get_index(rec.get("demographics"), self.demo_index, self.max_demographics)
        vec: List[float] = []
        vec.extend(_one_hot(c_idx, self.max_causes))
        vec.extend(_one_hot(d_idx, self.max_demographics))
        vec.append(_scale(rec.get("profitability", 0.0), _DEFAULT_BOUNDS["profitability"]))
        vec.append(_scale(rec.get("retention", 0.0), _DEFAULT_BOUNDS["retention"]))
        vec.append(_scale(rec.get("cac", 0.0), _DEFAULT_BOUNDS["cac"]))
        vec.append(_scale(rec.get("roi", 0.0), _DEFAULT_BOUNDS["roi"]))
        return vec

__all__ = ["FailureVectorizer"]
