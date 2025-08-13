"""Mixin adding vector embedding search to SQLite-backed databases.

This mixin stores embeddings in an ``embeddings`` table and provides
FAISS or Annoy based vector search with pluggable backend selection.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class EmbeddableDBMixin:
    """Mixin providing vector storage and similarity search.

    Subclasses are expected to provide a ``self.conn`` attribute referencing
    an :class:`sqlite3.Connection`.
    """

    def __init__(
        self,
        *,
        vector_backend: str = "faiss",
        index_path: str | Path = "embeddings.index",
        embedding_version: int = 1,
    ) -> None:
        self.vector_backend = vector_backend if vector_backend in {"faiss", "annoy"} else "annoy"
        self.index_path = Path(index_path)
        self.embedding_version = embedding_version
        self._vector_index = None
        self._vector_dim = 0
        self._id_map: List[Any] = []
        self._id_map_path = self.index_path.with_suffix(".json")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings(
                record_id TEXT PRIMARY KEY,
                vector TEXT,
                created_at TEXT,
                embedding_version INTEGER,
                kind TEXT,
                source_id TEXT
            )
            """
        )
        self.conn.commit()
        self._load_index()

    # ------------------------------------------------------------------
    # internal helpers
    def _load_index(self) -> None:
        """Load existing index and id map from disk if available."""
        if self._id_map_path.exists():
            try:
                self._id_map = json.loads(self._id_map_path.read_text())
            except Exception as exc:  # pragma: no cover - corrupted file
                logger.error("failed loading id map: %s", exc)
                self._id_map = []
        if not self.index_path.exists():
            return
        if self.vector_backend == "faiss":
            try:  # pragma: no cover - optional dependency
                import faiss  # type: ignore
            except Exception as exc:  # pragma: no cover
                logger.warning("FAISS backend unavailable: %s", exc)
                return
            self._vector_index = faiss.read_index(str(self.index_path))
            self._vector_dim = self._vector_index.d
        else:  # Annoy backend
            try:  # pragma: no cover - optional dependency
                from annoy import AnnoyIndex  # type: ignore
            except Exception as exc:  # pragma: no cover
                logger.warning("Annoy backend unavailable: %s", exc)
                return
            if self._id_map:
                row = self.conn.execute(
                    "SELECT vector FROM embeddings LIMIT 1"
                ).fetchone()
                if row:
                    self._vector_dim = len(json.loads(row[0]))
            if self._vector_dim:
                self._vector_index = AnnoyIndex(self._vector_dim, "angular")
                self._vector_index.load(str(self.index_path))

    def _ensure_index(self, dim: int) -> None:
        if self._vector_index is not None:
            return
        self._vector_dim = dim
        if self.vector_backend == "faiss":
            try:  # pragma: no cover - optional dependency
                import faiss  # type: ignore
            except Exception as exc:  # pragma: no cover
                logger.warning("FAISS backend unavailable: %s", exc)
                return
            self._vector_index = faiss.IndexFlatL2(dim)
        else:
            try:  # pragma: no cover - optional dependency
                from annoy import AnnoyIndex  # type: ignore
            except Exception as exc:  # pragma: no cover
                logger.warning("Annoy backend unavailable: %s", exc)
                return
            self._vector_index = AnnoyIndex(dim, "angular")

    def _save_index(self) -> None:
        if self._vector_index is None:
            return
        if self.vector_backend == "faiss":
            try:  # pragma: no cover - optional dependency
                import faiss  # type: ignore
            except Exception as exc:  # pragma: no cover
                logger.warning("FAISS backend unavailable: %s", exc)
                return
            faiss.write_index(self._vector_index, str(self.index_path))
        else:
            self._vector_index.save(str(self.index_path))
        self._id_map_path.write_text(json.dumps(self._id_map))

    # ------------------------------------------------------------------
    # public API
    def vector(self, record: Any) -> List[float] | None:
        """Return the stored embedding vector for ``record``.

        ``record`` may be a record object with an ``id`` attribute, a mapping
        containing ``id`` or the raw identifier itself.
        """

        if isinstance(record, (str, int)):
            record_id = record
        else:
            record_id = getattr(record, "id", None) or record.get("id")
        if record_id is None:
            return None
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE record_id=?",
            (str(record_id),),
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def add_embedding(
        self,
        record_id: Any,
        vector: Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store ``vector`` for ``record_id`` and add it to the search index."""

        metadata = metadata or {}
        created_at = metadata.get("created_at", datetime.utcnow().isoformat())
        kind = metadata.get("kind", "")
        source_id = metadata.get("source_id", "")
        version = metadata.get("embedding_version", self.embedding_version)
        vec_json = json.dumps(list(vector))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings(
                record_id, vector, created_at, embedding_version, kind, source_id
            ) VALUES(?,?,?,?,?,?)
            """,
            (str(record_id), vec_json, created_at, version, kind, source_id),
        )
        self.conn.commit()

        self._ensure_index(len(vector))
        if self._vector_index is None:
            return
        if self.vector_backend == "faiss":
            import numpy as np  # type: ignore

            self._vector_index.add(np.array([vector], dtype="float32"))
            self._id_map.append(record_id)
            self._save_index()
        else:
            from annoy import AnnoyIndex  # type: ignore

            self._vector_index = AnnoyIndex(len(vector), "angular")
            cur = self.conn.execute(
                "SELECT record_id, vector FROM embeddings ORDER BY rowid"
            )
            self._id_map = []
            for i, (rid, vec_json) in enumerate(cur.fetchall()):
                self._vector_index.add_item(i, json.loads(vec_json))
                self._id_map.append(rid)
            self._vector_index.build(10)
            self._save_index()

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 5
    ) -> List[Tuple[Any, float]]:
        """Search embeddings similar to ``vector``.

        Returns a list of ``(record_id, distance)`` tuples.
        """

        if self._vector_index is None:
            self._load_index()
        if self._vector_index is None:
            return []
        top_k = max(1, top_k)
        if self.vector_backend == "faiss":
            import numpy as np  # type: ignore

            dists, idxs = self._vector_index.search(
                np.array([vector], dtype="float32"), top_k
            )
            return [
                (self._id_map[i], float(d))
                for i, d in zip(idxs[0], dists[0])
                if i < len(self._id_map)
            ]
        ids, dists = self._vector_index.get_nns_by_vector(  # type: ignore
            list(vector), top_k, include_distances=True
        )
        return [
            (self._id_map[i], float(d))
            for i, d in zip(ids, dists)
            if i < len(self._id_map)
        ]
