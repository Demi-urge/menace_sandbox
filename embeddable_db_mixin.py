"""Mixin providing embedding storage and vector search backends.

This module offers :class:`EmbeddableDBMixin` which can be mixed into a
class managing a SQLite database.  The mixin stores embedding vectors in an
Annoy or FAISS index on disk and keeps companion metadata in a JSON file.  A
 lazily loaded `SentenceTransformer` model is provided for text-to-vector
encoding, allowing subclasses to embed arbitrary records.

Subclasses must provide a ``self.conn`` database connection and override
:meth:`vector` to return an embedding for a record.  To support
:meth:`backfill_embeddings`, subclasses should also implement
:meth:`iter_records` yielding ``(record_id, record, kind)`` tuples.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple
from time import perf_counter
import json
import logging
from security.secret_redactor import redact

# Lightweight license detection based on simple fingerprints.  This avoids
# embedding content that is under GPL or non‑commercial restrictions.
from legal.license_fingerprint import (
    LicenseType,
    detect_license,
    fingerprint as license_fingerprint,
)

try:  # pragma: no cover - optional dependency
    from annoy import AnnoyIndex
except Exception:  # pragma: no cover - Annoy not installed
    AnnoyIndex = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - FAISS not installed
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - NumPy not installed
    np = None  # type: ignore

try:
    from .vector_metrics_db import VectorMetricsDB
    from .embedding_stats_db import EmbeddingStatsDB
    from .metrics_exporter import (
        embedding_tokens_total as _EMBED_TOKENS,
        embedding_wall_seconds_total as _EMBED_WALL_TOTAL,
        embedding_store_seconds_total as _EMBED_STORE_TOTAL,
        embedding_stale_cost_seconds as _EMBED_STALE,
        embedding_wall_time_seconds as _EMBED_WALL_LAST,
        embedding_store_latency_seconds as _EMBED_STORE_LAST,
    )
except Exception:  # pragma: no cover - fallback to absolute imports
    from vector_metrics_db import VectorMetricsDB  # type: ignore
    from embedding_stats_db import EmbeddingStatsDB  # type: ignore
    from metrics_exporter import (
        embedding_tokens_total as _EMBED_TOKENS,
        embedding_wall_seconds_total as _EMBED_WALL_TOTAL,
        embedding_store_seconds_total as _EMBED_STORE_TOTAL,
        embedding_stale_cost_seconds as _EMBED_STALE,
        embedding_wall_time_seconds as _EMBED_WALL_LAST,
        embedding_store_latency_seconds as _EMBED_STORE_LAST,
    )

logger = logging.getLogger(__name__)


_VEC_METRICS = VectorMetricsDB()
_EMBED_STATS_DB = EmbeddingStatsDB("metrics.db")


def log_embedding_metrics(
    db_name: str,
    tokens: int,
    wall_time: float,
    store_latency: float,
    *,
    vector_id: str = "",
) -> None:
    """Log embedding metrics to Prometheus and persistent storage."""

    try:
        _EMBED_TOKENS.inc(tokens)
        _EMBED_WALL_TOTAL.inc(wall_time)
        _EMBED_STORE_TOTAL.inc(store_latency)
        _EMBED_WALL_LAST.set(wall_time)
        _EMBED_STORE_LAST.set(store_latency)
    except Exception:  # pragma: no cover - best effort
        pass
    try:
        _EMBED_STATS_DB.log(
            db_name=db_name,
            tokens=tokens,
            wall_ms=wall_time * 1000,
            store_ms=store_latency * 1000,
        )
        _VEC_METRICS.log_embedding(
            db=db_name,
            tokens=tokens,
            wall_time_ms=wall_time * 1000,
            store_time_ms=store_latency * 1000,
            vector_id=vector_id,
        )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to persist embedding metrics")


class EmbeddableDBMixin:
    """Add embedding storage and similarity search to a database class."""

    def __init__(
        self,
        *,
        index_path: str | Path = "embeddings.ann",
        metadata_path: str | Path = "embeddings.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_version: int = 1,
        backend: str = "annoy",
    ) -> None:
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        if metadata_path.name == "embeddings.json" and index_path.name != "embeddings.ann":
            # Derive metadata file alongside the provided index path to avoid
            # cross-database interference when tests supply unique index files
            metadata_path = index_path.with_suffix(".json")
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.embedding_version = embedding_version
        self.backend = backend

        self._model = None
        self._index: Any | None = None
        self._vector_dim = 0
        self._id_map: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._last_embedding_tokens = 0
        self._last_embedding_time = 0.0

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

        text = redact(text)

        start = perf_counter()
        tokens = 0
        try:  # pragma: no cover - optional dependency
            tokenizer = getattr(self.model, "tokenizer", None)
            if tokenizer:
                tokens = len(tokenizer.encode(text))
        except Exception:
            tokens = 0
        vec = self.model.encode(text).tolist()  # pragma: no cover - heavy
        self._last_embedding_tokens = tokens
        self._last_embedding_time = perf_counter() - start
        return vec

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

    def license_text(self, record: Any) -> str | None:
        """Return textual content to scan for license violations.

        Subclasses can override this to extract text from structured
        records. By default, if ``record`` is a string it is returned as-is
        otherwise ``None`` is returned, skipping the license check.
        """

        return record if isinstance(record, str) else None

    # ------------------------------------------------------------------
    # index persistence
    def load_index(self) -> None:
        """Load the vector index and metadata from disk if available."""
        if self.metadata_path.exists():
            data = json.loads(self.metadata_path.read_text())
            self._id_map = data.get("id_map", [])
            self._metadata = data.get("metadata", {})
            self._vector_dim = data.get("vector_dim", 0)
            if not self._id_map:
                self._id_map = list(self._metadata.keys())
        if self.backend == "annoy":
            if AnnoyIndex and self.index_path.exists() and self._vector_dim:
                self._index = AnnoyIndex(self._vector_dim, "angular")
                self._index.load(str(self.index_path))
            elif AnnoyIndex and self._metadata:
                self._rebuild_index()
        elif self.backend == "faiss":
            if faiss and self.index_path.exists() and self._vector_dim:
                self._index = faiss.read_index(str(self.index_path))
            elif faiss and self._metadata:
                self._rebuild_index()

    def save_index(self) -> None:
        """Persist vector index and metadata to disk."""
        if self._index is None:
            return
        if self.backend == "annoy":
            if not AnnoyIndex:
                return
            self._index.save(str(self.index_path))
        elif self.backend == "faiss":
            if not faiss:
                return
            faiss.write_index(self._index, str(self.index_path))
        data = {
            "id_map": self._id_map,
            "metadata": self._metadata,
            "vector_dim": self._vector_dim,
        }
        self.metadata_path.write_text(json.dumps(data, indent=2))

    def _rebuild_index(self) -> None:
        """Rebuild vector index from stored metadata."""
        if not self._metadata:
            self._index = None
            return
        self._vector_dim = len(next(iter(self._metadata.values()))["vector"])
        if self.backend == "annoy":
            if not AnnoyIndex:
                self._index = None
                return
            self._index = AnnoyIndex(self._vector_dim, "angular")
            for i, rid in enumerate(self._id_map):
                vec = self._metadata[rid]["vector"]
                self._index.add_item(i, vec)
            self._index.build(10)
        elif self.backend == "faiss":
            if not faiss or not np:
                self._index = None
                return
            self._index = faiss.IndexFlatIP(self._vector_dim)
            vectors = [self._metadata[rid]["vector"] for rid in self._id_map]
            if vectors:
                arr = np.array(vectors, dtype="float32")
                self._index.add(arr)

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
        text = self.license_text(record)
        if text:
            lic = detect_license(text)
            if lic is not LicenseType.UNKNOWN:
                try:  # pragma: no cover - best effort
                    hash_ = license_fingerprint(text)
                    log_fn = getattr(self, "log_license_violation", None)
                    if callable(log_fn):
                        log_fn("", lic.value, hash_)
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to log license violation for %s", record_id
                    )
                log_embedding_metrics(
                    self.__class__.__name__, 0, 0.0, 0.0, vector_id=str(record_id)
                )
                logger.warning(
                    "skipping embedding for %s due to license %s", record_id, lic.value
                )
                return
        record = redact(record) if isinstance(record, str) else record

        start = perf_counter()
        vec = self.vector(record)
        wall_time = perf_counter() - start
        tokens = getattr(self, "_last_embedding_tokens", 0)
        if not tokens and isinstance(record, str):  # pragma: no cover - best effort
            try:
                tokenizer = getattr(self.model, "tokenizer", None)
                if tokenizer:
                    tokens = len(tokenizer.encode(record))
            except Exception:  # pragma: no cover - best effort
                tokens = 0
        self._last_embedding_tokens = tokens
        self._last_embedding_time = wall_time

        rid = str(record_id)
        if rid not in self._metadata:
            self._id_map.append(rid)
        self._metadata[rid] = {
            "vector": list(vec),
            "created_at": datetime.utcnow().isoformat(),
            "embedding_version": self.embedding_version,
            "kind": kind,
            "source_id": source_id,
            "redacted": True,
        }
        self._rebuild_index()
        save_start = perf_counter()
        self.save_index()
        index_latency = perf_counter() - save_start
        log_embedding_metrics(
            self.__class__.__name__,
            tokens,
            wall_time,
            index_latency,
            vector_id=str(record_id),
        )

    def get_vector(self, record_id: Any) -> List[float] | None:
        """Return the stored embedding vector for ``record_id`` if present."""

        meta = self._metadata.get(str(record_id))
        if meta:
            self._record_staleness(str(record_id), meta.get("created_at"))
            return list(meta["vector"])
        return None

    def try_add_embedding(
        self,
        record_id: Any,
        record: Any,
        kind: str,
        *,
        source_id: str = "",
    ) -> None:
        """Best-effort variant of :meth:`add_embedding`."""

        try:
            self.add_embedding(record_id, record, kind, source_id=source_id)
        except Exception:  # pragma: no cover - best effort
            logging.exception("embedding hook failed for %s", record_id)

    def update_embedding_version(
        self, record_id: Any, *, embedding_version: int | None = None
    ) -> None:
        """Update ``embedding_version`` metadata for ``record_id``."""

        rid = str(record_id)
        if rid not in self._metadata:
            return
        version = (
            embedding_version if embedding_version is not None else self.embedding_version
        )
        self._metadata[rid]["embedding_version"] = int(version)
        self._metadata[rid]["created_at"] = datetime.utcnow().isoformat()
        self.save_index()

    def update_embedding_versions(
        self, record_ids: Sequence[Any], *, embedding_version: int | None = None
    ) -> None:
        """Bulk update ``embedding_version`` for multiple records."""

        version = (
            embedding_version if embedding_version is not None else self.embedding_version
        )
        now = datetime.utcnow().isoformat()
        updated = False
        for rid in map(str, record_ids):
            if rid in self._metadata:
                self._metadata[rid]["embedding_version"] = int(version)
                self._metadata[rid]["created_at"] = now
                updated = True
        if updated:
            self.save_index()

    # internal ---------------------------------------------------------
    def _record_staleness(self, rid: str, created_at: str | None) -> None:
        """Log how stale an embedding is when accessed."""
        if not created_at:
            return
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(created_at)).total_seconds()
        except Exception:
            return
        origin = getattr(self, "origin_db", self.__class__.__name__)
        if _EMBED_STALE:
            try:
                _EMBED_STALE.labels(origin_db=origin).set(age)
            except ValueError:  # pragma: no cover - labels not configured
                _EMBED_STALE.set(age)
        try:
            MetricsDB().log_embedding_staleness(origin, rid, age)
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist embedding staleness")

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """Return ``top_k`` records most similar to ``vector``."""

        if self._index is None:
            self.load_index()
        if self._index is None:
            return []
        if self.backend == "annoy":
            ids, dists = self._index.get_nns_by_vector(
                list(vector), top_k, include_distances=True
            )
            results: List[Tuple[Any, float]] = []
            for i, d in zip(ids, dists):
                if i < len(self._id_map):
                    rid = self._id_map[i]
                    meta = self._metadata.get(rid)
                    if not meta or not meta.get("redacted"):
                        continue
                    results.append((rid, float(d)))
                    self._record_staleness(rid, meta.get("created_at"))
            return results
        elif self.backend == "faiss":
            if not faiss or not np:
                return []
            vec = np.array([list(vector)], dtype="float32")
            dists, ids = self._index.search(vec, top_k)
            results: List[Tuple[Any, float]] = []
            for idx, dist in zip(ids[0], dists[0]):
                if 0 <= idx < len(self._id_map):
                    rid = self._id_map[idx]
                    meta = self._metadata.get(rid)
                    if not meta or not meta.get("redacted"):
                        continue
                    results.append((rid, float(dist)))
                    self._record_staleness(rid, meta.get("created_at"))
            return results
        return []

    def backfill_embeddings(self) -> None:
        """Generate embeddings for all records lacking them."""

        for record_id, record, kind in self.iter_records():
            if str(record_id) not in self._metadata:
                record = redact(record) if isinstance(record, str) else record
                self.add_embedding(record_id, record, kind)
