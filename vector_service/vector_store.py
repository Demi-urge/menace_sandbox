"""Vector store abstraction with multiple backends.

Currently supports FAISS, Annoy, Qdrant and Chroma.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Protocol

import json
import math
import uuid
from dynamic_path_router import resolve_path

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
        self.path = resolve_path(self.path)
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
        self.path = resolve_path(self.path)
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
    if backend == "faiss" and faiss is not None:
        return FaissVectorStore(dim=dim, path=Path(path))
    if backend == "qdrant" and QdrantClient is not None:
        return QdrantVectorStore(dim=dim, path=Path(path))
    if backend == "chroma" and chromadb is not None:
        return ChromaVectorStore(dim=dim, path=Path(path))
    return AnnoyVectorStore(dim=dim, path=Path(path), metric=metric)


_default_store: VectorStore | None = None
_stack_store: VectorStore | None = None
_stack_metadata_path: Path | None = None


def _coerce_path(value: Any) -> Path | None:
    if value in {None, "", b""}:
        return None
    try:
        path = Path(str(value))
    except Exception:
        return None
    try:
        return resolve_path(path)
    except Exception:
        return path.resolve() if path.is_absolute() else path


def _stack_vector_store_config() -> Tuple[int, Path, str, str, Path | None]:
    """Return configuration for the dedicated Stack vector index.

    The helper inspects the global configuration object for optional
    ``stack_vector_store`` and ``stack_dataset`` sections.  When settings are
    absent sensible defaults are used so tests can rely on ephemeral indexes.
    """

    dim = 384
    backend = "annoy"
    metric = "angular"
    index_path: Path | None = None
    metadata_path: Path | None = None

    try:  # pragma: no cover - configuration may be missing in tests
        from config import CONFIG

        vec_cfg = getattr(CONFIG, "vector", None)
        if vec_cfg is not None:
            try:
                dim = int(getattr(vec_cfg, "dimensions", dim))
            except Exception:
                pass

        stack_store_cfg = getattr(CONFIG, "stack_vector_store", None)
        if stack_store_cfg is None:
            stack_store_cfg = getattr(CONFIG, "vector_store", None)

        if stack_store_cfg is not None:
            try:
                backend = str(getattr(stack_store_cfg, "backend", backend))
            except Exception:
                pass
            try:
                metric = str(getattr(stack_store_cfg, "metric", metric))
            except Exception:
                pass
            index_candidate = None
            try:
                index_candidate = getattr(stack_store_cfg, "path")
            except AttributeError:
                index_candidate = None
            if index_candidate:
                coerced = _coerce_path(index_candidate)
                if coerced is not None:
                    index_path = coerced
            meta_candidate = None
            try:
                meta_candidate = getattr(stack_store_cfg, "metadata_path")
            except AttributeError:
                meta_candidate = None
            if meta_candidate:
                coerced = _coerce_path(meta_candidate)
                if coerced is not None:
                    metadata_path = coerced

        ctx_builder_cfg = getattr(CONFIG, "context_builder", None)
        if ctx_builder_cfg is not None:
            try:
                candidate = getattr(ctx_builder_cfg, "stack_index_path")
            except AttributeError:
                candidate = None
            if index_path is None and candidate:
                coerced = _coerce_path(candidate)
                if coerced is not None:
                    index_path = coerced
            try:
                candidate = getattr(ctx_builder_cfg, "stack_metadata_path")
            except AttributeError:
                candidate = None
            if metadata_path is None and candidate:
                coerced = _coerce_path(candidate)
                if coerced is not None:
                    metadata_path = coerced

        stack_dataset_cfg = getattr(CONFIG, "stack_dataset", None)
        if stack_dataset_cfg is not None:
            for attr in ("index_path", "vector_index_path"):
                try:
                    value = getattr(stack_dataset_cfg, attr)
                except AttributeError:
                    value = None
                if value:
                    coerced = _coerce_path(value)
                    if coerced is not None:
                        index_path = coerced
                        break
            if metadata_path is None:
                for attr in ("metadata_path", "metadata_db", "db_path"):
                    try:
                        value = getattr(stack_dataset_cfg, attr)
                    except AttributeError:
                        value = None
                    if value:
                        coerced = _coerce_path(value)
                        if coerced is not None:
                            metadata_path = coerced
                            break
            if index_path is None and metadata_path is not None:
                index_path = metadata_path.with_suffix(".index")
    except Exception:
        pass

    if index_path is None:
        index_path = _coerce_path("stack_embeddings.index") or Path(
            "stack_embeddings.index"
        )
    if metadata_path is None:
        metadata_path = _coerce_path("stack_embeddings.db")

    return dim, index_path, backend, metric, metadata_path


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


def get_stack_vector_store() -> VectorStore | None:
    """Return a cached :class:`VectorStore` dedicated to Stack embeddings."""

    global _stack_store, _stack_metadata_path
    if _stack_store is not None:
        return _stack_store

    dim, index_path, backend, metric, metadata_path = _stack_vector_store_config()
    _stack_metadata_path = metadata_path
    try:
        _stack_store = create_vector_store(dim, index_path, backend=backend, metric=metric)
    except Exception:
        _stack_store = None
    return _stack_store


def get_stack_metadata_path() -> Path | None:
    """Return the resolved metadata database path for Stack embeddings."""

    global _stack_store, _stack_metadata_path
    if _stack_store is None:
        get_stack_vector_store()
    return _stack_metadata_path


__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "AnnoyVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "create_vector_store",
    "get_default_vector_store",
    "get_stack_vector_store",
    "get_stack_metadata_path",
]
