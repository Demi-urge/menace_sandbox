from __future__ import annotations
"""Vectoriser for information records.

Required fields::
    content: str - full text body
    summary: str - brief description
"""
from dataclasses import dataclass
from typing import Any, Dict, List

_DEFAULT_BOUNDS = {"content_len": 10_000.0, "summary_len": 1_000.0}

def _scale(value: Any, bound: float) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    f = max(-bound, min(bound, f))
    return f / bound if bound else 0.0

@dataclass
class InformationVectorizer:
    """Simple length based embedding for information records."""

    def transform(self, info: Dict[str, Any]) -> List[float]:
        content = info.get("content", "")
        summary = info.get("summary", "")
        vec = [
            _scale(len(str(content)), _DEFAULT_BOUNDS["content_len"]),
            _scale(len(str(summary)), _DEFAULT_BOUNDS["summary_len"]),
        ]
        return vec

__all__ = ["InformationVectorizer"]
