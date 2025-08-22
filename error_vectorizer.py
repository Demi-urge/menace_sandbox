from __future__ import annotations

"""Vectoriser for error telemetry records."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from vector_utils import persist_embedding

_DEFAULT_BOUNDS = {"stack_len": 200.0}

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
class ErrorVectorizer:
    """Basic vectoriser for error log entries from ``errors.db``."""

    max_categories: int = 20
    max_modules: int = 50
    category_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    module_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})

    def fit(self, errors: Iterable[Dict[str, Any]]) -> "ErrorVectorizer":
        for err in errors:
            _get_index(err.get("category") or err.get("error_type"), self.category_index, self.max_categories)
            _get_index(err.get("root_module") or err.get("module"), self.module_index, self.max_modules)
        return self

    @property
    def dim(self) -> int:
        return self.max_categories + self.max_modules + 1

    def transform(self, err: Dict[str, Any]) -> List[float]:
        c_idx = _get_index(err.get("category") or err.get("error_type"), self.category_index, self.max_categories)
        m_idx = _get_index(err.get("root_module") or err.get("module"), self.module_index, self.max_modules)
        stack = err.get("stack_trace") or ""
        vec: List[float] = []
        vec.extend(_one_hot(c_idx, self.max_categories))
        vec.extend(_one_hot(m_idx, self.max_modules))
        vec.append(_scale(len(str(stack).splitlines()), _DEFAULT_BOUNDS["stack_len"]))
        return vec


_DEFAULT_VECTORIZER = ErrorVectorizer()


def vectorize_and_store(
    record_id: str,
    error: Dict[str, Any],
    *,
    path: str = "embeddings.jsonl",
    origin_db: str = "error",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``error`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(error)
    try:
        persist_embedding(
            "error",
            record_id,
            vec,
            path=path,
            origin_db=origin_db,
            metadata=metadata or {},
        )
    except TypeError:  # pragma: no cover - compatibility with older signatures
        persist_embedding("error", record_id, vec, path=path)
    return vec


__all__ = ["ErrorVectorizer", "vectorize_and_store"]
