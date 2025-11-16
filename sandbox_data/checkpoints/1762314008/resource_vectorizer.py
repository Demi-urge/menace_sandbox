from __future__ import annotations

"""Vectoriser for resource allocation and ROI history records."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List

_DEFAULT_BOUNDS = {
    "roi": 1000.0,
    "allocation": 1_000_000.0,
    "trend": 1000.0,
    "timestamp": 10_000_000_000.0,
}

def _scale(value: Any, bound: float) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    f = max(-bound, min(bound, f))
    return f / bound if bound else 0.0


def _parse_ts(ts: Any) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts).timestamp()
        except Exception:
            return 0.0
    return 0.0


@dataclass
class ResourceVectorizer:
    """Lightweight vectoriser for ``ROIHistoryDB`` records."""

    def fit(self, records: Iterable[Dict[str, Any]]) -> "ResourceVectorizer":
        return self

    @property
    def dim(self) -> int:
        return 4

    def transform(self, rec: Dict[str, Any]) -> List[float]:
        roi = _scale(rec.get("roi", 0.0), _DEFAULT_BOUNDS["roi"])
        alloc = _scale(rec.get("allocation", 0.0), _DEFAULT_BOUNDS["allocation"])
        trend = _scale(rec.get("roi_trend", rec.get("trend", 0.0)), _DEFAULT_BOUNDS["trend"])
        ts_val = _parse_ts(rec.get("ts") or rec.get("timestamp"))
        ts = _scale(ts_val, _DEFAULT_BOUNDS["timestamp"])
        return [roi, alloc, trend, ts]

__all__ = ["ResourceVectorizer"]
