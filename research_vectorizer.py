from __future__ import annotations

"""Vectoriser for research aggregation records."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

_DEFAULT_BOUNDS = {
    "data_depth": 100.0,
    "energy": 1_000.0,
    "corroboration": 100.0,
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
class ResearchVectorizer:
    """Simplistic vectoriser for research info records."""

    max_categories: int = 20
    max_types: int = 20
    category_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    type_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})

    def fit(self, records: Iterable[Dict[str, Any]]) -> "ResearchVectorizer":
        for rec in records:
            _get_index(rec.get("category"), self.category_index, self.max_categories)
            _get_index(rec.get("type"), self.type_index, self.max_types)
        return self

    @property
    def dim(self) -> int:
        return self.max_categories + self.max_types + 3

    def transform(self, rec: Dict[str, Any]) -> List[float]:
        c_idx = _get_index(rec.get("category"), self.category_index, self.max_categories)
        t_idx = _get_index(rec.get("type"), self.type_index, self.max_types)
        vec: List[float] = []
        vec.extend(_one_hot(c_idx, self.max_categories))
        vec.extend(_one_hot(t_idx, self.max_types))
        vec.append(_scale(rec.get("data_depth", 0.0), _DEFAULT_BOUNDS["data_depth"]))
        vec.append(_scale(rec.get("energy", 0.0), _DEFAULT_BOUNDS["energy"]))
        vec.append(_scale(rec.get("corroboration_count", 0.0), _DEFAULT_BOUNDS["corroboration"]))
        return vec

__all__ = ["ResearchVectorizer"]
