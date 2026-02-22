from __future__ import annotations
"""Vectoriser for discrepancy reports.

Required fields::
    description: str - explanation of the discrepancy
    severity: float - numeric severity level
"""
from dataclasses import dataclass
from typing import Any, Dict, List

_DEFAULT_BOUNDS = {"severity": 10.0, "desc_len": 1_000.0}

def _scale(value: Any, bound: float) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    f = max(-bound, min(bound, f))
    return f / bound if bound else 0.0

@dataclass
class DiscrepancyVectorizer:
    """Embeds severity and description length for discrepancy records."""

    def transform(self, rec: Dict[str, Any]) -> List[float]:
        vec = [
            _scale(rec.get("severity", 0.0), _DEFAULT_BOUNDS["severity"]),
            _scale(len(str(rec.get("description", ""))), _DEFAULT_BOUNDS["desc_len"]),
        ]
        return vec

__all__ = ["DiscrepancyVectorizer"]
