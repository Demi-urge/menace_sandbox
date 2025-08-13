"""Mixin providing embedding storage and Annoy based vector search.

This module offers :class:`EmbeddableDBMixin` which can be mixed into a
class managing a SQLite database.  The mixin stores embedding vectors in an
Annoy index on disk and keeps companion metadata in a JSON file.  A lazily
loaded `SentenceTransformer` model is provided for text-to-vector encoding,
allowing subclasses to embed arbitrary records.

Subclasses must provide a ``self.conn`` database connection and override
:meth:`vector` to return an embedding for a record.  To support
:meth:`backfill_embeddings`, subclasses should also implement
:meth:`iter_records` yielding ``(record_id, record, kind)`` tuples.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple
import json

try:  # pragma: no cover - optional dependency
    from annoy import AnnoyIndex
except Exception:  # pragma: no cover - Annoy not installed
    AnnoyIndex = None  # type: ignore


class EmbeddableDBMixin:
    """Add embedding storage and similarity search to a database class."""

    def __init__(
        self,
        *,
        index_path: str | Path = "embeddings.ann",
        metadata_path: str | Path = "embeddings.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_version: int = 1,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.embedding_version = embedding_version

        self._model = None
        self._index: AnnoyIndex | None = None
        self._vector_dim = 0
        self._id_map: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}

        self.load_index()

    # ------------------------------------------------------------------
    # model helpers
    @property
    def model(self):
        """Lazily loaded `SentenceTransformer` instance."""
        if self._model is None:  # pragma: no cover - heavy dependency
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_text(self, text: str) -> List[float]:
        """Encode ``text`` using the SentenceTransformer model."""
        return self.model.encode(text).tolist()  # pragma: no cover - heavy

    # ------------------------------------------------------------------
    # methods expected to be overridden
    def vector(self, record: Any) -> List[float]:
        """Return an embedding vector for ``record``.

        Subclasses **must** override this method.
        """

        raise NotImplementedError

    def iter_records(self) -> Iterator[Tuple[Any, Any, str]]:
        """Yield ``(record_id, record, kind)`` tuples for backfilling.

        Override in subclasses that use :meth:`backfill_embeddings`.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # index persistence
    def load_index(self) -> None:
        """Load Annoy index and metadata from disk if available."""
        if self.metadata_path.exists():
            data = json.loads(self.metadata_path.read_text())
            self._id_map = data.get("id_map", [])
            self._metadata = data.get("metadata", {})
            self._vector_dim = data.get("vector_dim", 0)
            if not self._id_map:
                self._id_map = list(self._metadata.keys())
        if AnnoyIndex and self.index_path.exists() and self._vector_dim:
            self._index = AnnoyIndex(self._vector_dim, "angular")
            self._index.load(str(self.index_path))
        elif AnnoyIndex and self._metadata:
            self._rebuild_index()

    def save_index(self) -> None:
        """Persist Annoy index and metadata to disk."""
        if not AnnoyIndex or self._index is None:
            return
        self._index.save(str(self.index_path))
        data = {
            "id_map": self._id_map,
            "metadata": self._metadata,
            "vector_dim": self._vector_dim,
        }
        self.metadata_path.write_text(json.dumps(data, indent=2))

    def _rebuild_index(self) -> None:
        """Rebuild Annoy index from stored metadata."""
        if not AnnoyIndex or not self._metadata:
            self._index = None
            return
        self._vector_dim = len(next(iter(self._metadata.values()))["vector"])
        self._index = AnnoyIndex(self._vector_dim, "angular")
        for i, rid in enumerate(self._id_map):
            vec = self._metadata[rid]["vector"]
            self._index.add_item(i, vec)
        self._index.build(10)

    # ------------------------------------------------------------------
    # public API
    def add_embedding(
        self,
        record_id: Any,
        record: Any,
        kind: str,
        *,
        source_id: str = "",
    ) -> None:
        """Embed ``record`` and store the vector and metadata."""

        vec = self.vector(record)
        rid = str(record_id)
        if rid not in self._metadata:
            self._id_map.append(rid)
        self._metadata[rid] = {
            "vector": list(vec),
            "created_at": datetime.utcnow().isoformat(),
            "embedding_version": self.embedding_version,
            "kind": kind,
            "source_id": source_id,
        }
        self._rebuild_index()
        self.save_index()

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """Return ``top_k`` records most similar to ``vector``."""

        if self._index is None:
            self.load_index()
        if self._index is None:
            return []
        ids, dists = self._index.get_nns_by_vector(
            list(vector), top_k, include_distances=True
        )
        return [
            (self._id_map[i], float(d))
            for i, d in zip(ids, dists)
            if i < len(self._id_map)
        ]

    def backfill_embeddings(self) -> None:
        """Generate embeddings for all records lacking them."""

        for record_id, record, kind in self.iter_records():
            if str(record_id) not in self._metadata:
                self.add_embedding(record_id, record, kind)
