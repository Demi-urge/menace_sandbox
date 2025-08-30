"""Vector store abstraction with FAISS and Annoy backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Protocol

import json
import math

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback when module missing
    faiss = None  # type: ignore

try:  # Annoy has a local fallback implementation in this repo
    from annoy import AnnoyIndex  # type: ignore
except Exception:  # pragma: no cover
    AnnoyIndex = None  # type: ignore

import numpy as np


class VectorStore(Protocol):
    """Simple vector storage and retrieval interface."""

    def add(
        self,
        kind: str,
        record_id: str,
        vector: Sequence[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist ``vector`` for ``record_id`` of type ``kind``."""

    def query(self, vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Return ``top_k`` nearest neighbour ids with their distances."""

    def load(self) -> None:
        """Load any previously persisted data from disk."""


# ---------------------------------------------------------------------------
# FAISS implementation
# ---------------------------------------------------------------------------


@dataclass
class FaissVectorStore:
    dim: int
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.meta_path = self.path.with_suffix(".meta.json")
        if self.path.exists():
            self.load()
        else:
            if faiss is None:
                raise RuntimeError("faiss backend requested but faiss not available")
            self.index = faiss.IndexFlatL2(self.dim)
            self.ids: List[str] = []
            self.meta: List[Dict[str, Any]] = []

    def add(
        self,
        kind: str,
        record_id: str,
        vector: Sequence[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if faiss is None:
            raise RuntimeError("faiss not available")
        vec = np.array([vector], dtype="float32")
        self.index.add(vec)
        self.ids.append(record_id)
        self.meta.append(
            {
                "type": kind,
                "id": record_id,
                "origin_db": origin_db,
                "metadata": dict(metadata or {}),
            }
        )
        self._save()

    def query(self, vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.ids:
            return []
        vec = np.array([vector], dtype="float32")
        distances, indices = self.index.search(vec, top_k)
        result: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.ids):
                result.append((self.ids[idx], float(dist)))
        return result

    def load(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss not available")
        self.index = faiss.read_index(str(self.path))
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.ids = list(map(str, data.get("ids", [])))
            self.meta = list(data.get("meta", []))
        else:
            self.ids = []
            self.meta = []

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.path))
        with self.meta_path.open("w", encoding="utf-8") as fh:
            json.dump({"ids": self.ids, "meta": self.meta}, fh)


# ---------------------------------------------------------------------------
# Annoy implementation
# ---------------------------------------------------------------------------


@dataclass
class AnnoyVectorStore:
    dim: int
    path: Path
    metric: str = "angular"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.index = AnnoyIndex(self.dim, self.metric)
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []
        self.meta: List[Dict[str, Any]] = []
        self._built = False
        if self.path.exists():
            self.load()

    def add(
        self,
        kind: str,
        record_id: str,
        vector: Sequence[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        idx = len(self.ids)
        self.index.add_item(idx, list(vector))
        self.ids.append(record_id)
        self.vectors.append(list(vector))
        self.meta.append(
            {
                "type": kind,
                "id": record_id,
                "origin_db": origin_db,
                "metadata": dict(metadata or {}),
            }
        )
        self._built = False
        self._save()

    def query(self, vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.ids:
            return []
        if not self._built:
            self.index.build(10)
            self._built = True
        idxs = self.index.get_nns_by_vector(list(vector), top_k)
        result: List[Tuple[str, float]] = []
        for idx in idxs:
            if 0 <= idx < len(self.ids):
                dist = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(vector, self.vectors[idx]))
                )
                result.append((self.ids[idx], dist))
        return result

    def load(self) -> None:
        self.index.load(str(self.path))
        self._built = True
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.ids = list(map(str, data.get("ids", [])))
            self.meta = list(data.get("meta", []))
            self.vectors = [list(map(float, v)) for v in data.get("vectors", [])]
        else:
            self.ids = []
            self.meta = []
            self.vectors = []

    def _save(self) -> None:
        if not self._built:
            self.index.build(10)
            self._built = True
        self.index.save(str(self.path))
        with self.meta_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {"ids": self.ids, "meta": self.meta, "vectors": self.vectors}, fh
            )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_vector_store(
    dim: int,
    path: str | Path,
    *,
    backend: str | None = None,
    metric: str = "angular",
) -> VectorStore:
    """Create a ``VectorStore`` instance for the given configuration."""

    backend = (backend or "faiss").lower()
    if backend == "faiss" and faiss is not None:
        return FaissVectorStore(dim=dim, path=Path(path))
    return AnnoyVectorStore(dim=dim, path=Path(path), metric=metric)


_default_store: VectorStore | None = None


def get_default_vector_store() -> VectorStore | None:
    """Return a cached ``VectorStore`` based on global configuration."""

    global _default_store
    if _default_store is not None:
        return _default_store

    try:  # pragma: no cover - configuration optional in tests
        from config import CONFIG

        cfg = getattr(CONFIG, "vector_store", None)
        vec_cfg = getattr(CONFIG, "vector", None)
        if cfg is None or vec_cfg is None:
            return None
        _default_store = create_vector_store(
            dim=vec_cfg.dimensions,
            path=cfg.path,
            backend=cfg.backend,
            metric="angular",
        )
    except Exception:
        return None
    return _default_store


__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "AnnoyVectorStore",
    "create_vector_store",
    "get_default_vector_store",
]
