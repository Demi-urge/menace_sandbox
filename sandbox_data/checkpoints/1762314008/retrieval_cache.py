"""SQLite-backed cache for retrieval results with TTL eviction."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, List

from db_router import GLOBAL_ROUTER, init_db_router

MENACE_ID = "retrieval_cache"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)

__all__ = ["RetrievalCache"]


@dataclass
class RetrievalCache:
    """Persist semantic retrieval results on disk.

    The cache is keyed by the search ``query`` and a canonicalised chain of
    database names.  Entries expire automatically after ``ttl`` seconds.
    """

    path: str | Path = "metrics.db"
    ttl: int = 3600
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.router = DB_ROUTER
        self._conn = self.router.get_connection("retrieval_cache")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_cache(
                query TEXT NOT NULL,
                db_chain TEXT NOT NULL,
                ts REAL NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(query, db_chain)
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    def _key(self, query: str, dbs: Sequence[str] | None) -> tuple[str, str]:
        db_chain = "|".join(dbs or [])
        return query, db_chain

    # ------------------------------------------------------------------
    def get(self, query: str, dbs: Sequence[str] | None) -> List[dict[str, Any]] | None:
        """Return cached results for ``query``/``dbs`` if present and fresh."""

        q, db_chain = self._key(query, dbs)
        row = self._conn.execute(
            "SELECT payload, ts FROM retrieval_cache WHERE query=? AND db_chain=?",
            (q, db_chain),
        ).fetchone()
        if not row:
            return None
        payload, ts = row
        if self.ttl and time.time() - ts > self.ttl:
            self._conn.execute(
                "DELETE FROM retrieval_cache WHERE query=? AND db_chain=?",
                (q, db_chain),
            )
            self._conn.commit()
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def set(
        self, query: str, dbs: Sequence[str] | None, results: List[dict[str, Any]]
    ) -> None:
        """Store ``results`` for the given ``query``/``dbs`` combination."""

        q, db_chain = self._key(query, dbs)
        payload = json.dumps(results)
        self._conn.execute(
            "REPLACE INTO retrieval_cache(query, db_chain, ts, payload) VALUES (?,?,?,?)",
            (q, db_chain, time.time(), payload),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all cached entries."""

        self._conn.execute("DELETE FROM retrieval_cache")
        self._conn.commit()

