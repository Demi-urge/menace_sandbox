"""SQLite-backed storage for module intent embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
import sqlite3

from module_synergy_grapher import ModuleSynergyGrapher
from vector_service import EmbeddableDBMixin
try:  # pragma: no cover - import available when running inside package
    from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
except Exception:  # pragma: no cover - top level import fallback
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router  # type: ignore

try:  # pragma: no cover - support package and flat layouts
    from .dynamic_path_router import resolve_path, resolve_module_path
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path, resolve_module_path  # type: ignore

from intent_vectorizer import IntentVectorizer


@dataclass
class IntentRecord:
    """Representation of a module tracked for intent embedding."""

    path: str
    added: str = datetime.utcnow().isoformat()
    id: int = 0


class IntentDB(EmbeddableDBMixin):
    """Persist intent embeddings for repository modules."""

    DB_FILE = "intent.db"

    def __init__(
        self,
        path: str | Path = "intent.db",
        *,
        router: DBRouter | None = None,
        vector_index_path: str | Path | None = None,
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router("intent", str(path), str(path))
        self.conn = self.router.get_connection("intent")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intent_modules(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                added TEXT,
                source_menace_id TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intent_path ON intent_modules(path)"
        )
        self.conn.commit()
        index_path = (
            Path(vector_index_path)
            if vector_index_path is not None
            else Path(path).with_suffix(".index")
        )
        meta_path = index_path.with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )
        self._vectorizer = IntentVectorizer()

    # ------------------------------------------------------------------
    def add(self, path: str) -> int:
        """Insert ``path`` into the database if not present."""

        cur = self.conn.execute(
            "INSERT OR IGNORE INTO intent_modules(path, added, source_menace_id) VALUES (?,?,?)",
            (path, datetime.utcnow().isoformat(), self.router.menace_id),
        )
        self.conn.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.conn.execute("SELECT id FROM intent_modules WHERE path=?", (path,)).fetchone()
        return int(row["id"]) if row else 0

    # ------------------------------------------------------------------
    def iter_records(self) -> Iterator[Tuple[int, Dict[str, Any], str]]:
        """Yield records for embedding backfills."""

        for row in self.conn.execute("SELECT id, path FROM intent_modules"):
            yield int(row["id"]), {"path": row["path"]}, "module"

    # ------------------------------------------------------------------
    def _bundle(self, path: str) -> str:
        return self._vectorizer.bundle(path)

    def license_text(self, rec: Any) -> str | None:  # pragma: no cover - simple wrapper
        path = None
        if isinstance(rec, dict):
            path = rec.get("path")
        elif isinstance(rec, (str, Path)):
            path = str(rec)
        elif isinstance(rec, int):
            row = self.conn.execute("SELECT path FROM intent_modules WHERE id=?", (rec,)).fetchone()
            path = row["path"] if row else None
        if not path:
            return None
        return self._bundle(path)

    # ------------------------------------------------------------------
    def vector(self, rec: Any) -> List[float]:
        path = None
        text = None
        if isinstance(rec, dict):
            path = rec.get("path")
            text = rec.get("text")
        elif isinstance(rec, (str, Path)):
            path = str(rec)
        elif isinstance(rec, int):
            row = self.conn.execute("SELECT path FROM intent_modules WHERE id=?", (rec,)).fetchone()
            path = row["path"] if row else None
        if text is None and path:
            text = self._bundle(path)
        if not text:
            return []
        return self.encode_text(text)

    # ------------------------------------------------------------------
    def index_synergy_cluster(self, module_name: str, threshold: float) -> int | None:
        """Index an aggregated embedding for the synergy cluster around ``module_name``.

        Each member module of the cluster is embedded individually and the
        resulting vectors are averaged to form the cluster representation.
        The combined vector is stored in the same index as regular modules so
        retrieval can surface cluster level results alongside individual files.
        """

        grapher = ModuleSynergyGrapher()
        cluster = grapher.get_synergy_cluster(module_name, threshold)
        if not cluster:
            return None

        vectors: List[List[float]] = []
        members: List[str] = []
        for mod in cluster:
            path = resolve_module_path(mod.replace("/", "."))
            try:
                text = self._vectorizer.bundle(path)
            except Exception:
                text = ""
            if not text:
                continue
            vec = self.encode_text(text)
            if vec:
                vectors.append(vec)
                members.append(str(path))
        if not vectors:
            return None

        dim = len(vectors[0])
        agg = [0.0] * dim
        for vec in vectors:
            if len(vec) != dim:
                continue
            for i, val in enumerate(vec):
                agg[i] += float(val)
        avg_vec = [v / len(vectors) for v in agg]

        key = f"cluster:{module_name}"
        rec_id = self.add(key)
        rid = str(rec_id)
        if rid not in self._metadata:
            self._id_map.append(rid)
        self._metadata[rid] = {
            "vector": list(avg_vec),
            "created_at": datetime.utcnow().isoformat(),
            "embedding_version": self.embedding_version,
            "kind": "cluster",
            "source_id": "",
            "redacted": True,
            "members": sorted(members),
        }
        self._rebuild_index()
        self.save_index()
        return rec_id


__all__ = ["IntentRecord", "IntentDB"]
