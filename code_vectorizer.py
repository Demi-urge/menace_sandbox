from __future__ import annotations
"""Vectoriser for source code snippets.

Required fields::
    language: str - programming language of the snippet
    content: str - source code text
"""
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List

_DEFAULT_BOUNDS = {"lines": 1_000.0}


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


logger = logging.getLogger(__name__)


def _scale(value: Any, bound: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Value could not be converted to float for scaling",
            extra={"value": value},
            exc_info=exc,
        )
        return 0.0
    except Exception:
        logger.exception(
            "Unexpected error during value scaling", extra={"value": value}
        )
        raise
    f = max(-bound, min(bound, f))
    return f / bound if bound else 0.0


@dataclass
class CodeVectorizer:
    """Encodes language and basic size features for code snippets."""

    max_languages: int = 10
    language_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})

    def transform(self, rec: Dict[str, Any]) -> List[float]:
        idx = _get_index(rec.get("language"), self.language_index, self.max_languages)
        lines = len(str(rec.get("content", "")).splitlines())
        vec: List[float] = []
        vec.extend(_one_hot(idx, self.max_languages))
        vec.append(_scale(lines, _DEFAULT_BOUNDS["lines"]))
        return vec


__all__ = ["CodeVectorizer"]
