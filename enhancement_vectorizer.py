from __future__ import annotations

"""Vectoriser for enhancement records."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from governed_embeddings import governed_embed
from vector_utils import persist_embedding

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - sentence-transformers optional
    SentenceTransformer = None  # type: ignore

_DEFAULT_BOUNDS = {
    "score": 100.0,
    "cost_estimate": 1_000_000.0,
    "num_tags": 20.0,
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
class EnhancementVectorizer:
    """Simple vectoriser for entries from ``chatgpt_enhancement_bot``."""

    max_types: int = 20
    max_categories: int = 20
    type_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    category_index: Dict[str, int] = field(default_factory=lambda: {"other": 0})
    _model: SentenceTransformer | None = field(default=None, init=False, repr=False)

    def fit(self, enhancements: Iterable[Dict[str, Any]]) -> "EnhancementVectorizer":
        for enh in enhancements:
            _get_index(enh.get("type") or enh.get("type_"), self.type_index, self.max_types)
            _get_index(enh.get("category"), self.category_index, self.max_categories)
        return self

    @property
    def dim(self) -> int:
        embed_dim = 384 if SentenceTransformer is not None else 0
        return self.max_types + self.max_categories + 3 + embed_dim

    def transform(self, enh: Dict[str, Any]) -> List[float]:
        t_idx = _get_index(enh.get("type") or enh.get("type_"), self.type_index, self.max_types)
        c_idx = _get_index(enh.get("category"), self.category_index, self.max_categories)
        tags = enh.get("tags") or []
        vec: List[float] = []
        vec.extend(_one_hot(t_idx, self.max_types))
        vec.extend(_one_hot(c_idx, self.max_categories))
        vec.append(_scale(enh.get("score", 0.0), _DEFAULT_BOUNDS["score"]))
        vec.append(_scale(enh.get("cost_estimate", 0.0), _DEFAULT_BOUNDS["cost_estimate"]))
        vec.append(_scale(len(tags), _DEFAULT_BOUNDS["num_tags"]))
        desc = str(enh.get("description", ""))
        if SentenceTransformer is not None:
            if self._model is None:
                from huggingface_hub import login
                import os

                login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = governed_embed(desc, self._model) or []
            vec.extend(float(x) for x in emb)
        return vec


_DEFAULT_VECTORIZER = EnhancementVectorizer()


def vectorize_and_store(
    record_id: str,
    enhancement: Dict[str, Any],
    *,
    path: str = "embeddings.jsonl",
    origin_db: str = "enhancement",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``enhancement`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(enhancement)
    try:
        persist_embedding(
            "enhancement",
            record_id,
            vec,
            path=path,
            origin_db=origin_db,
            metadata=metadata or {},
        )
    except TypeError:  # pragma: no cover - compatibility with older signatures
        persist_embedding("enhancement", record_id, vec, path=path)
    return vec


__all__ = ["EnhancementVectorizer", "vectorize_and_store"]
