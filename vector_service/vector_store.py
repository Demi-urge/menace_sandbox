"""Vector store abstraction with multiple backends.

Currently supports FAISS, Annoy, Qdrant and Chroma.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Protocol

import json
import logging
import math
import os
import uuid
from dynamic_path_router import get_project_root, resolve_path

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback when module missing
    faiss = None  # type: ignore

try:  # Annoy has a local fallback implementation in this repo
    from annoy import AnnoyIndex  # type: ignore
except Exception:  # pragma: no cover
    AnnoyIndex = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http.models import Distance, PointStruct, VectorParams  # type: ignore
except Exception:  # pragma: no cover - fallback when module missing
    QdrantClient = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - fallback when module missing
    chromadb = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback when module missing
    np = None  # type: ignore


logger = logging.getLogger(__name__)


def _trace(event: str, **extra: Any) -> None:
    flag = os.getenv("VECTOR_SERVICE_TRACE", "")
    if flag and flag.lower() not in {"0", "false", "no", "off"}:
        payload = {"event": event, **extra}
        logger.log(logging.INFO, "vector-store: %s", event, extra=payload)


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


def _resolve_or_prepare_path(path: Path | str) -> Path:
    """Return an absolute path, creating parents when missing."""

    candidate = Path(path)
    try:
        return resolve_path(candidate.as_posix())
    except FileNotFoundError:
        if not candidate.is_absolute():
            try:
                base = get_project_root()
            except Exception:
                base = Path.cwd()
            candidate = base / candidate
        candidate = candidate.expanduser()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate


@dataclass
class FaissVectorStore:
    dim: int
    path: Path

    def __post_init__(self) -> None:
        self.path = _resolve_or_prepare_path(self.path)
        self.meta_path = self.path.with_suffix(".meta.json")
        if self.path.exists():
            self.load()
        else:
            if faiss is None:
                raise RuntimeError("faiss backend requested but faiss not available")
            if np is None:
                raise RuntimeError("faiss backend requires numpy, which is missing")
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
        if np is None:
            raise RuntimeError("numpy is required for faiss vector store operations")
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
        if np is None:
            raise RuntimeError("numpy is required for faiss vector store operations")
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
        if AnnoyIndex is None:
            raise RuntimeError("annoy backend requested but annoy is not available")
        self.path = _resolve_or_prepare_path(self.path)
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


@dataclass
class InMemoryVectorStore:
    """Minimal vector store fallback with JSON persistence."""

    dim: int
    path: Path
    metric: str = "euclidean"

    def __post_init__(self) -> None:
        self.path = _resolve_or_prepare_path(self.path)
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []
        self.meta: List[Dict[str, Any]] = []
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
        self.ids.append(record_id)
        self.vectors.append([float(v) for v in vector])
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
        query_vec = [float(v) for v in vector]
        distances = []
        for idx, candidate in enumerate(self.vectors):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(query_vec, candidate)))
            distances.append((self.ids[idx], dist))
        distances.sort(key=lambda item: item[1])
        return distances[:top_k]

    def load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            self.ids, self.vectors, self.meta = [], [], []
            return
        self.ids = [str(v) for v in payload.get("ids", [])]
        self.vectors = [list(map(float, v)) for v in payload.get("vectors", [])]
        self.meta = list(payload.get("meta", []))

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump({"ids": self.ids, "vectors": self.vectors, "meta": self.meta}, fh)


# ---------------------------------------------------------------------------
# Qdrant implementation
# ---------------------------------------------------------------------------


@dataclass
class QdrantVectorStore:
    dim: int
    path: Path
    collection_name: str = "vectors"

    def __post_init__(self) -> None:
        if QdrantClient is None:  # pragma: no cover - dependency missing
            raise RuntimeError("qdrant backend requested but qdrant-client not available")
        self.path = Path(self.path)
        self.client = QdrantClient(path=str(self.path))
        self._ensure_collection()

    # -- internal helpers -------------------------------------------------
    def _ensure_collection(self) -> None:
        assert QdrantClient is not None  # for type checkers
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    # -- VectorStore interface -------------------------------------------
    def add(
        self,
        kind: str,
        record_id: str,
        vector: Sequence[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        qid = uuid.uuid5(uuid.NAMESPACE_OID, record_id).hex
        payload = {
            "type": kind,
            "id": record_id,
            "origin_db": origin_db,
            "metadata": dict(metadata or {}),
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=qid, vector=list(vector), payload=payload)],
        )

    def query(self, vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if QdrantClient is None:
            raise RuntimeError("qdrant-client not available")
        res = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(vector),
            limit=top_k,
        )
        result: List[Tuple[str, float]] = []
        for r in res:
            rid = r.payload.get("id") if isinstance(r.payload, dict) else None
            result.append((str(rid or r.id), float(r.score)))
        return result

    def load(self) -> None:
        if QdrantClient is None:
            raise RuntimeError("qdrant-client not available")
        self.client = QdrantClient(path=str(self.path))
        self._ensure_collection()


# ---------------------------------------------------------------------------
# Chroma implementation
# ---------------------------------------------------------------------------


@dataclass
class ChromaVectorStore:
    dim: int
    path: Path
    collection_name: str = "vectors"

    def __post_init__(self) -> None:
        if chromadb is None:  # pragma: no cover - dependency missing
            raise RuntimeError("chroma backend requested but chromadb not available")
        self.path = Path(self.path)
        self._connect()

    def _connect(self) -> None:
        assert chromadb is not None  # for type checking
        client = chromadb.PersistentClient(path=str(self.path))
        self.client = client
        self.collection = client.get_or_create_collection(self.collection_name)

    def add(
        self,
        kind: str,
        record_id: str,
        vector: Sequence[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        meta: Dict[str, Any] = {
            "type": kind,
            "id": record_id,
        }
        if origin_db is not None:
            meta["origin_db"] = origin_db
        if metadata:
            meta["metadata"] = json.dumps(dict(metadata))
        self.collection.add(ids=[record_id], embeddings=[list(vector)], metadatas=[meta])

    def query(self, vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[list(vector)],
            n_results=top_k,
            include=["distances"],
        )
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [(str(i), float(d)) for i, d in zip(ids, dists)]

    def load(self) -> None:
        if chromadb is None:
            raise RuntimeError("chromadb not available")
        self._connect()


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
    if backend == "faiss" and faiss is not None and np is not None:
        return FaissVectorStore(dim=dim, path=Path(path))
    if backend == "qdrant" and QdrantClient is not None:
        return QdrantVectorStore(dim=dim, path=Path(path))
    if backend == "chroma" and chromadb is not None:
        return ChromaVectorStore(dim=dim, path=Path(path))
    if backend == "annoy" and AnnoyIndex is not None:
        return AnnoyVectorStore(dim=dim, path=Path(path), metric=metric)
    if AnnoyIndex is not None:
        return AnnoyVectorStore(dim=dim, path=Path(path), metric=metric)
    _trace("vector_store.fallback", backend="inmemory", requested=backend)
    return InMemoryVectorStore(dim=dim, path=Path(path), metric=metric)


_default_store: VectorStore | None = None


def get_default_vector_store() -> VectorStore | None:
    """Return a cached ``VectorStore`` based on global configuration."""

    global _default_store
    if _default_store is not None:
        _trace("vector_store.default.cached", backend=_default_store.__class__.__name__)
        return _default_store

    try:  # pragma: no cover - configuration optional in tests
        from config import CONFIG

        cfg = getattr(CONFIG, "vector_store", None)
        vec_cfg = getattr(CONFIG, "vector", None)
        if cfg is None or vec_cfg is None:
            _trace(
                "vector_store.default.config_missing",
                has_vector_store_cfg=cfg is not None,
                has_vector_cfg=vec_cfg is not None,
            )
            return None
        _trace(
            "vector_store.default.create",
            backend=getattr(cfg, "backend", None),
            path=getattr(cfg, "path", None),
            dimensions=getattr(vec_cfg, "dimensions", None),
        )
        _default_store = create_vector_store(
            dim=vec_cfg.dimensions,
            path=cfg.path,
            backend=cfg.backend,
            metric="angular",
        )
        _trace(
            "vector_store.default.created",
            backend=_default_store.__class__.__name__,
        )
    except Exception:
        _trace("vector_store.default.error", error="config_import_failed")
        return None
    return _default_store


__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "AnnoyVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "create_vector_store",
    "get_default_vector_store",
]
