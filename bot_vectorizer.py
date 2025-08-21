from __future__ import annotations

"""Feature vectorisation for bot definitions."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

_DEFAULT_BOUNDS = {"num_tasks": 20.0, "estimated_profit": 1_000_000.0}

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
class BotVectorizer:
    """Lightweight vectoriser for :mod:`bot_database` records."""

    max_types: int = 20
    max_status: int = 10
    type_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    status_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})

    def fit(self, bots: Iterable[Dict[str, Any]]) -> "BotVectorizer":
        for b in bots:
            _get_index(b.get("type") or b.get("type_"), self.type_index, self.max_types)
            _get_index(b.get("status"), self.status_index, self.max_status)
        return self

    @property
    def dim(self) -> int:
        return self.max_types + self.max_status + 2

    def transform(self, bot: Dict[str, Any]) -> List[float]:
        t_idx = _get_index(bot.get("type") or bot.get("type_"), self.type_index, self.max_types)
        s_idx = _get_index(bot.get("status"), self.status_index, self.max_status)
        vec: List[float] = []
        vec.extend(_one_hot(t_idx, self.max_types))
        vec.extend(_one_hot(s_idx, self.max_status))
        vec.append(_scale(len(bot.get("tasks", [])), _DEFAULT_BOUNDS["num_tasks"]))
        vec.append(_scale(bot.get("estimated_profit", 0.0), _DEFAULT_BOUNDS["estimated_profit"]))
        return vec

__all__ = ["BotVectorizer"]
