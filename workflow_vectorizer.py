from __future__ import annotations

"""Vectorisation utilities for workflow specifications."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from vector_utils import persist_embedding

_DEFAULT_BOUNDS = {
    "num_steps": 20.0,
    "duration": 10_000.0,
    "estimated_profit": 1_000_000.0,
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
class WorkflowVectorizer:
    """Lightweight vectoriser for records from ``workflows.db``.

    The vector uses fixed one-hot slots for categories and statuses. Unknown
    values extend the internal mappings until ``max_categories`` or
    ``max_status`` slots are filled. Additional unseen values fall back to the
    ``"other"`` slot at index 0. To support more distinct categories or
    statuses, increase the corresponding ``max_*`` parameter and recompute any
    stored embeddings so that downstream components see the new vector
    dimensionality.
    """

    max_categories: int = 20
    max_status: int = 10
    category_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    status_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})

    def fit(self, workflows: Iterable[Dict[str, Any]]) -> "WorkflowVectorizer":
        for wf in workflows:
            _get_index(wf.get("category"), self.category_index, self.max_categories)
            _get_index(wf.get("status"), self.status_index, self.max_status)
        return self

    @property
    def dim(self) -> int:
        return self.max_categories + self.max_status + 3

    def transform(self, wf: Dict[str, Any]) -> List[float]:
        c_idx = _get_index(wf.get("category"), self.category_index, self.max_categories)
        s_idx = _get_index(wf.get("status"), self.status_index, self.max_status)
        steps = wf.get("workflow") or wf.get("task_sequence") or []
        vec: List[float] = []
        vec.extend(_one_hot(c_idx, self.max_categories))
        vec.extend(_one_hot(s_idx, self.max_status))
        vec.append(_scale(len(steps), _DEFAULT_BOUNDS["num_steps"]))
        vec.append(_scale(wf.get("workflow_duration", 0.0), _DEFAULT_BOUNDS["duration"]))
        vec.append(
            _scale(
                wf.get("estimated_profit_per_bot", 0.0),
                _DEFAULT_BOUNDS["estimated_profit"],
            )
        )
        return vec


_DEFAULT_VECTORIZER = WorkflowVectorizer()


def vectorize_and_store(
    record_id: str,
    workflow: Dict[str, Any],
    *,
    path: str = "embeddings.jsonl",
    origin_db: str = "workflow",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``workflow`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(workflow)
    try:
        persist_embedding(
            "workflow",
            record_id,
            vec,
            path=path,
            origin_db=origin_db,
            metadata=metadata or {},
        )
    except TypeError:  # pragma: no cover - compatibility with older signatures
        persist_embedding("workflow", record_id, vec, path=path)
    return vec

__all__ = ["WorkflowVectorizer", "vectorize_and_store"]
