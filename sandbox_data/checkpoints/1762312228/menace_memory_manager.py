"""Centralised memory manager for Menace bots."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

from db_router import GLOBAL_ROUTER, init_db_router

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    KMeans = None  # type: ignore

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore

logger = logging.getLogger(__name__)


def _memory_embedder_timeout() -> float | None:
    """Return the timeout used when fetching the shared embedder.

    The timeout defaults to a small value so that repeated instantiations of
    :class:`MenaceMemoryManager` do not serially block sandbox start-up when the
    embedder is unavailable.  Users can opt-in to the legacy behaviour by
    setting ``MENACE_MEMORY_EMBEDDER_TIMEOUT`` to ``"default"`` which disables
    the override and lets :func:`governed_embeddings.get_embedder` use its
    global timeout.  Any negative values are treated as zero to avoid surprising
    wait times.
    """

    raw = os.getenv("MENACE_MEMORY_EMBEDDER_TIMEOUT", "").strip().lower()
    if not raw:
        return 5.0
    if raw in {"default", "none", "auto"}:
        return None
    try:
        timeout = float(raw)
    except Exception:
        logger.warning(
            "invalid MENACE_MEMORY_EMBEDDER_TIMEOUT=%r; defaulting to 5s", raw
        )
        return 5.0
    if timeout < 0:
        logger.warning(
            "MENACE_MEMORY_EMBEDDER_TIMEOUT must be non-negative; treating as 0s"
        )
        return 0.0
    return timeout


_MEMORY_EMBEDDER_TIMEOUT = _memory_embedder_timeout()


class _SimpleKMeans:
    """Fallback KMeans clustering for when scikit-learn is unavailable."""

    def __init__(self, n_clusters: int = 8, iters: int = 10) -> None:
        self.n_clusters = n_clusters
        self.iters = iters
        self.centers: List[List[float]] | None = None

    def fit(self, X: List[List[float]]) -> None:
        import random

        if not X:
            self.centers = []
            return
        self.centers = random.sample(X, min(self.n_clusters, len(X)))
        for _ in range(self.iters):
            clusters = [[] for _ in range(len(self.centers))]
            for vec in X:
                idx = self._closest(vec)[0]
                clusters[idx].append(vec)
            for i, cluster in enumerate(clusters):
                if cluster:
                    self.centers[i] = [sum(vals) / len(vals) for vals in zip(*cluster)]

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._closest(vec)[0] for vec in X]

    def _closest(self, vec: List[float]) -> tuple[int, float]:
        import math

        best = 0
        best_dist = float("inf")
        for i, c in enumerate(self.centers or []):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, c)))
            if dist < best_dist:
                best = i
                best_dist = dist
        return best, best_dist

try:  # optional dependency for embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional
    SentenceTransformer = None  # type: ignore

from governed_embeddings import governed_embed, get_embedder

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - circular imports
    from .bot_database import BotDB
    from .research_aggregator_bot import InfoDB
    from .databases import MenaceDB
else:  # pragma: no cover
    BotDB = InfoDB = MenaceDB = object  # type: ignore

try:  # pragma: no cover - support flat imports
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback when package not initialised
    from unified_event_bus import UnifiedEventBus  # type: ignore

try:
    from .knowledge_graph import KnowledgeGraph
except Exception:  # pragma: no cover - fallback for flat layout
    from knowledge_graph import KnowledgeGraph  # type: ignore
from gpt_memory_interface import GPTMemoryInterface

try:
    from .chatgpt_enhancement_bot import summarise_text as _summarise_text
except Exception:  # pragma: no cover - fallback summariser
    def _summarise_text(text: str, ratio: float = 0.2) -> str:
        text = text.strip()
        if not text:
            return ""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if len(sentences) <= 1:
            return text
        count = max(1, int(len(sentences) * ratio))
        return ". ".join(sentences[:count]) + "."


@dataclass
class MemoryEntry:
    key: str
    data: str
    version: int
    tags: str
    ts: str = datetime.utcnow().isoformat()


class MenaceMemoryManager(GPTMemoryInterface):
    """SQLite-backed manager that stores versioned data with tags."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        bot_db: "BotDB" | None = None,
        info_db: "InfoDB" | None = None,
        embedder: SentenceTransformer | None = None,
        menace_db: "MenaceDB" | None = None,
        cluster_backend: str = "hdbscan",
        recluster_interval: int = 100,
        vector_backend: str | None = None,
        vector_index_path: Path | str | None = None,
        summary_interval: int = 50,
    ) -> None:
        # allow connections to be shared across threads
        self._owns_connection = False
        if db_path is not None:
            target = Path(db_path)
            if str(target) != ":memory:":
                target.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(  # noqa: SQL001
                str(target), check_same_thread=False
            )
            self._owns_connection = True
        else:
            router = GLOBAL_ROUTER or init_db_router(
                os.getenv("MENACE_ID", "memory_manager")
            )
            with router.get_connection("memory") as conn:
                self.conn = conn
        self.subscribers: List[Callable[[MemoryEntry], None]] = []
        self.event_bus = event_bus
        self.bot_db = bot_db
        self.info_db = info_db
        self.menace_db = menace_db
        self.graph = KnowledgeGraph()
        if embedder is not None:
            self.embedder = embedder
        else:
            timeout = _MEMORY_EMBEDDER_TIMEOUT
            wait_for: float | None = None if timeout is None else max(0.0, timeout)
            self.embedder = get_embedder(timeout=wait_for)
            if self.embedder is None and wait_for not in (None, 0.0):
                logger.debug(
                    "MenaceMemoryManager continuing without embedder after %.1fs wait", wait_for
                )
        self.cluster_backend = cluster_backend if cluster_backend in {"hdbscan", "faiss"} else "hdbscan"
        self.recluster_interval = max(1, recluster_interval)
        self._log_count = 0
        self._faiss_index = None
        self.vector_backend = vector_backend or ""
        self.vector_index_path = Path(vector_index_path or "vector.index") if self.vector_backend else None
        self._vector_index = None
        self._vector_dim = 0
        self.has_fts = False
        self.summary_interval = max(0, summary_interval)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory(
                key TEXT,
                data TEXT,
                version INTEGER,
                tags TEXT,
                ts TEXT,
                bot_id INTEGER,
                info_id INTEGER
            )
            """
        )
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(key, data, tags, ts, version UNINDEXED)"
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO memory_fts(rowid, key, data, tags, ts, version) SELECT rowid, key, data, tags, ts, version FROM memory"
            )
            self.has_fts = True
        except sqlite3.OperationalError:
            self.has_fts = False
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_embeddings(rowid INTEGER PRIMARY KEY, embedding TEXT)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_clusters(
                cluster_id INTEGER,
                rowid INTEGER,
                PRIMARY KEY(cluster_id, rowid)
            )
            """
        )
        self.conn.commit()
        if self.vector_backend:
            try:
                self._load_vector_index()
                if self._vector_index is None:
                    self.migrate_embeddings_to_index()
            except Exception:
                self._vector_index = None

    def subscribe(self, callback: Callable[[MemoryEntry], None]) -> None:
        """Register a callback to receive new memory entries."""
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[MemoryEntry], None]) -> None:
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def _embed(self, text: str) -> Optional[List[float]]:
        return governed_embed(text, self.embedder)

    # ------------------------------------------------------------------
    def _load_vector_index(self) -> None:
        """Load persistent vector index if available."""
        if not self.vector_backend or not self.vector_index_path:
            return
        if self.vector_backend == "faiss_disk" and faiss:
            try:
                if self.vector_index_path.exists():
                    self._vector_index = faiss.read_index(str(self.vector_index_path))
                    self._vector_dim = self._vector_index.d
            except Exception:
                self._vector_index = None
        elif self.vector_backend == "annoy":
            try:
                from annoy import AnnoyIndex  # type: ignore
            except Exception:
                return
            meta = self.vector_index_path.with_suffix(self.vector_index_path.suffix + ".meta")
            if self.vector_index_path.exists() and meta.exists():
                try:
                    info = json.loads(meta.read_text())
                    dim = int(info.get("dim", 0))
                except Exception:
                    dim = 0
                if dim:
                    idx = AnnoyIndex(dim, "angular")
                    try:
                        idx.load(str(self.vector_index_path))
                        self._vector_index = idx
                        self._vector_dim = dim
                    except Exception:
                        self._vector_index = None

    def migrate_embeddings_to_index(self) -> int:
        """Rebuild persistent index from stored embeddings."""
        if not self.vector_backend or not self.vector_index_path:
            return 0
        cur = self.conn.execute("SELECT rowid, embedding FROM memory_embeddings")
        rows = cur.fetchall()
        vectors: list[list[float]] = []
        ids: list[int] = []
        for rid, emb_json in rows:
            try:
                vec = json.loads(emb_json)
            except Exception:
                continue
            ids.append(int(rid))
            vectors.append(vec)
        if not vectors:
            return 0
        dim = len(vectors[0])
        if self.vector_backend == "faiss_disk" and faiss:
            import numpy as np
            arr = np.array(vectors, dtype="float32")
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
            index.add_with_ids(arr, np.array(ids, dtype="int64"))
            faiss.write_index(index, str(self.vector_index_path))
            self._vector_index = index
            self._vector_dim = dim
        elif self.vector_backend == "annoy":
            try:
                from annoy import AnnoyIndex  # type: ignore
            except Exception:
                return 0
            index = AnnoyIndex(dim, "angular")
            for rid, vec in zip(ids, vectors):
                index.add_item(int(rid), vec)
            index.build(10)
            index.save(str(self.vector_index_path))
            meta = self.vector_index_path.with_suffix(self.vector_index_path.suffix + ".meta")
            meta.write_text(json.dumps({"dim": dim}))
            self._vector_index = index
            self._vector_dim = dim
        return len(vectors)

    def _index_add(self, rowid: int, embedding: list[float]) -> None:
        """Add a new embedding to the persistent index."""
        if not self.vector_backend or not self.vector_index_path:
            return
        if self.vector_backend == "faiss_disk" and faiss:
            import numpy as np
            if self._vector_index is None:
                self._vector_dim = len(embedding)
                self._vector_index = faiss.IndexIDMap(faiss.IndexFlatL2(self._vector_dim))
            self._vector_index.add_with_ids(
                np.array([embedding], dtype="float32"),
                np.array([rowid], dtype="int64"),
            )
            try:
                faiss.write_index(self._vector_index, str(self.vector_index_path))
            except Exception:
                logger.exception(
                    "Failed to write faiss index to %s", self.vector_index_path
                )
        elif self.vector_backend == "annoy":
            try:
                from annoy import AnnoyIndex  # type: ignore
            except Exception:
                return
            self._vector_index = None
            self.migrate_embeddings_to_index()

    def store(
        self,
        key: str,
        data: str | dict,
        tags: str = "",
        *,
        bot_id: int | None = None,
        info_id: int | None = None,
    ) -> int:
        """Convenience helper to log a new versioned entry."""
        version = self.next_version(key)
        if isinstance(data, dict):
            import json

            data = json.dumps(data)
        entry = MemoryEntry(key, str(data), version, tags)
        self.log(entry, bot_id=bot_id, info_id=info_id)
        if self.event_bus:
            try:
                payload = {
                    "key": entry.key,
                    "data": entry.data,
                    "version": entry.version,
                    "tags": entry.tags,
                    "ts": entry.ts,
                    "bot_id": bot_id,
                    "info_id": info_id,
                }
                self.event_bus.publish("memory:new", payload)
            except Exception:
                logger.exception(
                    "Failed to publish memory:new event for %s", entry.key
                )
        return version

    def log(self, entry: MemoryEntry, *, bot_id: int | None = None, info_id: int | None = None) -> None:
        self.conn.execute(
            "INSERT INTO memory(key, data, version, tags, ts, bot_id, info_id) VALUES(?,?,?,?,?,?,?)",
            (entry.key, entry.data, entry.version, entry.tags, entry.ts, bot_id, info_id),
        )
        rowid = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        embedding = self._embed(entry.data)
        if embedding is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO memory_embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, json.dumps(embedding)),
            )
            self._index_add(rowid, embedding)
            cur = self.conn.execute(
                "SELECT DISTINCT cluster_id FROM memory_clusters"
            )
            row = cur.fetchone()
            if row:
                # assign to nearest cluster
                clusters = [int(r[0]) for r in self.conn.execute("SELECT DISTINCT cluster_id FROM memory_clusters").fetchall()]
                if clusters:
                    vectors = []
                    for cid in clusters:
                        r = self.conn.execute(
                            "SELECT embedding FROM memory_embeddings m JOIN memory_clusters c ON m.rowid=c.rowid WHERE c.cluster_id=? LIMIT 1",
                            (cid,),
                        ).fetchone()
                        if r and r[0]:
                            try:
                                vectors.append(json.loads(r[0]))
                            except Exception:
                                vectors.append(embedding)
                        else:
                            vectors.append(embedding)
                    km = _SimpleKMeans(n_clusters=len(clusters))
                    km.centers = vectors  # type: ignore[assignment]
                    cid, _ = km._closest(embedding)
                    cluster_id = clusters[cid]
                    self.conn.execute(
                        "INSERT INTO memory_clusters(cluster_id, rowid) VALUES (?, ?)",
                        (cluster_id, rowid),
                    )
            if self.menace_db:
                try:
                    with self.menace_db.engine.begin() as conn:
                        conn.execute(
                            self.menace_db.memory_embeddings.insert().values(
                                key=entry.key,
                                data=entry.data,
                                version=entry.version,
                                tags=entry.tags,
                                ts=entry.ts,
                                embedding=json.dumps(embedding),
                            )
                        )
                except Exception:
                    logger.exception("Failed to sync embedding to menace DB")
        if getattr(self, "has_fts", False):
            try:
                self.conn.execute(
                    "INSERT INTO memory_fts(rowid, key, data, tags, ts, version) VALUES((SELECT last_insert_rowid()),?,?,?,?,?)",
                    (entry.key, entry.data, entry.tags, entry.ts, entry.version),
                )
            except sqlite3.OperationalError:
                self.has_fts = False
        self.conn.commit()
        if self.graph:
            try:
                tag_list = [t.strip() for t in entry.tags.replace(",", " ").split() if t.strip()]
                self.graph.add_memory_entry(entry.key, tag_list)
                if any(t in {"improvement", "bugfix"} for t in tag_list):
                    bots = [t.split(":", 1)[1] for t in tag_list if t.startswith("bot:")]
                    codes = [t.split(":", 1)[1] for t in tag_list if t.startswith("code:")]
                    errs = [
                        t.split(":", 1)[1]
                        for t in tag_list
                        if t.startswith("error:") or t.startswith("error_category:")
                    ]
                    self.graph.add_gpt_insight(
                        entry.key,
                        bots=bots or None,
                        code_paths=codes or None,
                        error_categories=errs or None,
                    )
            except Exception:
                logger.exception(
                    "Failed to update knowledge graph for key %s", entry.key
                )
        for cb in list(self.subscribers):
            try:
                cb(entry)
            except Exception:
                logger.exception(
                    "Subscriber callback failed for key %s", entry.key
                )

        if (
            self.summary_interval
            and not entry.key.endswith(":summary")
            and entry.version % self.summary_interval == 0
        ):
            try:
                self.summarise_memory(entry.key, limit=self.summary_interval, condense=True)
            except Exception:
                logger.exception("automatic summarise failed for %s", entry.key)

        self._log_count += 1
        if self.cluster_backend == "faiss" and self._log_count % self.recluster_interval == 0:
            try:
                self.cluster_embeddings()
            except Exception:
                logger.exception("automatic recluster failed")

    def next_version(self, key: str) -> int:
        cur = self.conn.execute(
            "SELECT MAX(version) FROM memory WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return (row[0] or 0) + 1

    def query(self, key: str, limit: int = 1) -> List[MemoryEntry]:
        cur = self.conn.execute(
            "SELECT key, data, version, tags, ts FROM memory WHERE key = ? ORDER BY version DESC LIMIT ?",
            (key, limit),
        )
        rows = cur.fetchall()
        return [MemoryEntry(*r) for r in rows]

    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        if getattr(self, "has_fts", False):
            try:
                cur = self.conn.execute(
                    "SELECT key, data, version, tags, ts FROM memory_fts WHERE tags MATCH ?",
                    (f"{tag}*",),
                )
                rows = cur.fetchall()
                return [MemoryEntry(*r) for r in rows]
            except sqlite3.OperationalError:
                self.has_fts = False
        cur = self.conn.execute(
            "SELECT key, data, version, tags, ts FROM memory WHERE tags LIKE ?",
            (f"%{tag}%",),
        )
        rows = cur.fetchall()
        return [MemoryEntry(*r) for r in rows]

    def search(self, text: str, limit: int = 20) -> List[MemoryEntry]:
        """Full text search across keys and data if FTS is available."""
        if getattr(self, "has_fts", False):
            try:
                cur = self.conn.execute(
                    "SELECT key, data, version, tags, ts FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?",
                    (f"{text}*", limit),
                )
                rows = cur.fetchall()
                return [MemoryEntry(*r) for r in rows]
            except sqlite3.OperationalError:
                self.has_fts = False
        pattern = f"%{text}%"
        cur = self.conn.execute(
            "SELECT key, data, version, tags, ts FROM memory WHERE key LIKE ? OR data LIKE ? LIMIT ?",
            (pattern, pattern, limit),
        )
        rows = cur.fetchall()
        return [MemoryEntry(*r) for r in rows]

    def query_vector(self, text: str, limit: int = 5) -> List[MemoryEntry]:
        embedding = self._embed(text)
        if embedding is None:
            return self.search(text, limit)
        if self.vector_backend == "faiss_disk" and faiss and self._vector_index is not None:
            import numpy as np
            try:
                xq = np.array([embedding], dtype="float32")
                _d, idx = self._vector_index.search(xq, limit)
                ids = [int(i) for i in idx[0] if i != -1]
                if not ids:
                    return []
                placeholders = ",".join("?" for _ in ids)
                cur = self.conn.execute(
                    f"SELECT key, data, version, tags, ts FROM memory WHERE rowid IN ({placeholders})",
                    tuple(ids),
                )
                rows = cur.fetchall()
                return [MemoryEntry(*r) for r in rows]
            except Exception:
                logger.error("faiss_disk vector search failed", exc_info=True)
        elif self.vector_backend == "annoy" and self._vector_index is not None:
            try:
                ids = self._vector_index.get_nns_by_vector(embedding, limit)
                if not ids:
                    return []
                placeholders = ",".join("?" for _ in ids)
                cur = self.conn.execute(
                    f"SELECT key, data, version, tags, ts FROM memory WHERE rowid IN ({placeholders})",
                    tuple(ids),
                )
                rows = cur.fetchall()
                return [MemoryEntry(*r) for r in rows]
            except Exception:
                logger.error("annoy vector search failed", exc_info=True)
        if self.cluster_backend == "faiss" and faiss and self._faiss_index is not None:
            import numpy as np
            try:
                xq = np.array([embedding], dtype="float32")
                _d, idx = self._faiss_index.search(xq, limit)
                ids = [int(i) for i in idx[0] if i != -1]
                if not ids:
                    return []
                placeholders = ",".join("?" for _ in ids)
                cur = self.conn.execute(
                    f"SELECT key, data, version, tags, ts FROM memory WHERE rowid IN ({placeholders})",
                    tuple(ids),
                )
                rows = cur.fetchall()
                return [MemoryEntry(*r) for r in rows]
            except Exception:
                logger.error("faiss cluster search failed", exc_info=True)
        cur = self.conn.execute(
            "SELECT m.key, m.data, m.version, m.tags, m.ts, e.embedding FROM memory_embeddings e JOIN memory m ON m.rowid=e.rowid"
        )
        rows = cur.fetchall()
        scored: List[tuple[float, MemoryEntry]] = []
        for key, data, version, tags, ts, emb_json in rows:
            if not emb_json:
                continue
            try:
                emb = json.loads(emb_json)
            except Exception:
                continue
            num = sum(a * b for a, b in zip(embedding, emb))
            denom = (sum(a * a for a in embedding) ** 0.5) * (sum(b * b for b in emb) ** 0.5)
            score = num / denom if denom else 0.0
            scored.append((score, MemoryEntry(key, data, version, tags, ts)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def refresh_embeddings(self) -> int:
        """Generate embeddings for stored entries missing vectors."""
        cur = self.conn.execute(
            "SELECT rowid, data FROM memory WHERE rowid NOT IN (SELECT rowid FROM memory_embeddings)"
        )
        rows = cur.fetchall()
        count = 0
        for rowid, data in rows:
            emb = self._embed(data)
            if emb is None:
                continue
            self.conn.execute(
                "INSERT OR REPLACE INTO memory_embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, json.dumps(emb)),
            )
            count += 1
            if self.menace_db:
                try:
                    with self.menace_db.engine.begin() as conn:
                        conn.execute(
                            self.menace_db.memory_embeddings.insert().values(
                                key="", data=data, version=0, tags="", ts="", embedding=json.dumps(emb)
                            )
                        )
                except Exception:
                    logger.exception(
                        "Failed to store embedding in menace DB during refresh"
                    )
        self.conn.commit()

        if count:
            try:
                self.cluster_embeddings()
            except Exception:
                logger.exception("clustering embeddings failed during refresh")
        return count

    # --------------------------------------------------------------
    def summarise_memory(
        self,
        key: str,
        limit: int = 20,
        *,
        ratio: float = 0.2,
        store: bool = True,
        condense: bool = False,
    ) -> str:
        """Summarise recent entries for *key* using simple heuristics."""
        cur = self.conn.execute(
            "SELECT rowid, data, version FROM memory WHERE key=? ORDER BY version DESC LIMIT ?",
            (key, limit),
        )
        rows = cur.fetchall()
        if not rows:
            return ""
        rows.reverse()
        text = "\n".join(r[1] for r in rows)
        summary = _summarise_text(text, ratio=ratio)
        refs = f"{rows[0][2]}-{rows[-1][2]}"
        stored = False
        if store and summary:
            try:
                self.store(f"{key}:summary", summary, tags=f"summary refs={refs}")
                stored = True
            except Exception:
                logger.exception("Failed to store summary for %s", key)
        if condense and stored:
            rowids = [r[0] for r in rows]
            placeholders = ",".join("?" for _ in rowids)
            self.conn.execute(
                f"DELETE FROM memory WHERE rowid IN ({placeholders})",
                rowids,
            )
            self.conn.execute(
                f"DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})",
                rowids,
            )
            self.conn.execute(
                f"DELETE FROM memory_clusters WHERE rowid IN ({placeholders})",
                rowids,
            )
            if getattr(self, "has_fts", False):
                try:
                    self.conn.execute(
                        f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})",
                        rowids,
                    )
                except sqlite3.OperationalError:
                    self.has_fts = False
            self.conn.commit()
            if self.vector_backend:
                try:
                    self._vector_index = None
                    self.migrate_embeddings_to_index()
                except Exception:
                    logger.exception("Failed to rebuild vector index after condense")
            try:
                self.cluster_embeddings()
            except Exception:
                logger.exception("clustering embeddings failed after condense")
        return summary

    # ------------------------------------------------------------------
    def cluster_embeddings(self, n_clusters: int = 8, *, min_cluster_size: int = 5) -> int:
        """Cluster embeddings into groups for quick retrieval."""
        if self.vector_backend and self._vector_index is None:
            try:
                self.migrate_embeddings_to_index()
            except Exception:
                logger.exception("index migration failed during clustering")
        cur = self.conn.execute(
            "SELECT rowid, embedding FROM memory_embeddings"
        )
        rows = cur.fetchall()
        if not rows:
            return 0
        vectors = []
        rowids = []
        for rowid, emb_json in rows:
            try:
                emb = json.loads(emb_json)
            except Exception:
                continue
            vectors.append(emb)
            rowids.append(rowid)

        if self.cluster_backend == "faiss" and faiss:
            try:
                import numpy as np
            except Exception:
                self._faiss_index = object()
                return 0
            try:
                arr = np.array(vectors, dtype="float32")
                kmeans = faiss.Kmeans(len(vectors[0]), n_clusters, niter=20, verbose=False)
                kmeans.train(arr)
                _, res = kmeans.index.search(arr, 1)
                labels = [int(r[0]) for r in res]
            except Exception:
                labels = [0 for _ in rowids]
            try:
                index = faiss.IndexIDMap(faiss.IndexFlatL2(len(vectors[0])))
                index.add_with_ids(arr, np.array(rowids, dtype="int64"))
                self._faiss_index = index
            except Exception:
                self._faiss_index = None
        elif self.vector_backend == "faiss_disk" and faiss:
            try:
                import numpy as np
            except Exception:
                self._vector_index = object()
                return 0
            arr = np.array(vectors, dtype="float32")
            index = faiss.IndexIDMap(faiss.IndexFlatL2(len(vectors[0])))
            index.add_with_ids(arr, np.array(rowids, dtype="int64"))
            self._vector_index = index
            try:
                faiss.write_index(index, str(self.vector_index_path))
                self._vector_dim = len(vectors[0])
            except Exception:
                logger.exception("Failed to save faiss index during clustering")
            labels = [0 for _ in rowids]
        elif self.vector_backend == "annoy":
            try:
                from annoy import AnnoyIndex  # type: ignore
            except Exception:
                logger.exception("Annoy unavailable for clustering")
            else:
                index = AnnoyIndex(len(vectors[0]), "angular")
                for rid, vec in zip(rowids, vectors):
                    index.add_item(int(rid), vec)
                index.build(10)
                try:
                    index.save(str(self.vector_index_path))
                    meta = self.vector_index_path.with_suffix(self.vector_index_path.suffix + ".meta")
                    meta.write_text(json.dumps({"dim": len(vectors[0])}))
                    self._vector_index = index
                    self._vector_dim = len(vectors[0])
                except Exception:
                    logger.exception("Failed to save Annoy index")
            labels = [0 for _ in rowids]
        else:
            if hdbscan:
                try:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                    labels = clusterer.fit_predict(vectors)
                except Exception:
                    labels = []
            else:
                if KMeans:
                    km = KMeans(n_clusters=n_clusters, n_init='auto')  # type: ignore[arg-type]
                else:
                    km = _SimpleKMeans(n_clusters=n_clusters)
                km.fit(vectors)
                labels = km.predict(vectors)
        if not labels:
            return 0
        self.conn.execute("DELETE FROM memory_clusters")
        self.conn.executemany(
            "INSERT INTO memory_clusters(cluster_id, rowid) VALUES (?, ?)",
            [(int(lbl), int(rid)) for lbl, rid in zip(labels, rowids)],
        )
        self.conn.commit()
        return len(labels)

    def query_cluster(self, text: str, limit: int = 5) -> List[MemoryEntry]:
        """Return entries from the closest embedding cluster."""
        if self.cluster_backend == "faiss" and self._faiss_index is not None:
            return self.query_vector(text, limit)
        emb = self._embed(text)
        if emb is None:
            return self.search(text, limit)
        cur = self.conn.execute(
            "SELECT DISTINCT cluster_id FROM memory_clusters"
        )
        clusters = [int(r[0]) for r in cur.fetchall()]
        if not clusters:
            return self.query_vector(text, limit)
        import math
        best_cluster = 0
        best_dist = float('inf')
        for cid in clusters:
            cur = self.conn.execute(
                "SELECT embedding FROM memory_embeddings m JOIN memory_clusters c ON m.rowid=c.rowid WHERE c.cluster_id=? LIMIT 1",
                (cid,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                continue
            try:
                vec = json.loads(row[0])
            except Exception:
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, emb)))
            if dist < best_dist:
                best_cluster = cid
                best_dist = dist
        cur = self.conn.execute(
            "SELECT m.key, m.data, m.version, m.tags, m.ts FROM memory_clusters c JOIN memory m ON m.rowid=c.rowid WHERE c.cluster_id=?",
            (best_cluster,),
        )
        rows = cur.fetchall()
        result = [MemoryEntry(*r) for r in rows]
        result.sort(key=lambda e: e.ts, reverse=True)
        return result[:limit]

    # ------------------------------------------------------- unified interface
    def log_interaction(
        self, prompt: str, response: str, tags: Sequence[str] | None = None
    ) -> int:
        """Record a prompt/response pair using the generic store method."""

        payload = json.dumps({"prompt": prompt, "response": response})
        tag_str = ",".join(tags or [])
        return self.store(prompt, payload, tags=tag_str)

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
        **_: Any,
    ) -> List[MemoryEntry]:
        """Return memory entries matching ``query`` and optional ``tags``."""

        entries = self.search(query, limit * 5 if tags else limit)
        if tags:
            wanted = set(tags)
            entries = [
                e
                for e in entries
                if not wanted.isdisjoint(t for t in e.tags.split(",") if t)
            ]
        return entries[:limit]

    def retrieve(
        self, query: str, limit: int = 5, tags: Sequence[str] | None = None
    ) -> List[MemoryEntry]:
        """Alias for :meth:`search_context` for interface compatibility."""

        return self.search_context(query, limit=limit, tags=tags)

__all__ = [
    "MemoryEntry",
    "MenaceMemoryManager",
    "_SimpleKMeans",
]
