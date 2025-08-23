from __future__ import annotations

"""SQLite logger for embedding statistics."""

from datetime import datetime
from pathlib import Path
import sqlite3

from db_router import GLOBAL_ROUTER, init_db_router

MENACE_ID = "embedding_stats_db"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)


class EmbeddingStatsDB:
    """Persist high-level embedding metrics.

    Each record captures the timestamp, database name, token usage and timing
    information for a single embedding operation.  The database is lazily
    initialised on first use and is safe for use across threads.
    """

    def __init__(self, path: str | Path = "metrics.db") -> None:
        """Initialise the metrics table via ``DBRouter``."""

        self.router = DB_ROUTER
        self.conn = self.router.get_connection("embedding_stats")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_stats(
                db_name TEXT,
                tokens INTEGER,
                wall_ms REAL,
                store_ms REAL,
                patch_id TEXT,
                db_source TEXT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def log(
        self,
        db_name: str,
        tokens: int,
        wall_ms: float,
        store_ms: float,
        *,
        patch_id: str = "",
        db_source: str = "",
    ) -> None:
        """Insert a new embedding statistics row."""

        ts = datetime.utcnow().isoformat()
        conn = self.router.get_connection("embedding_stats")
        conn.execute(
            """
            INSERT INTO embedding_stats(
                db_name, tokens, wall_ms, store_ms, patch_id, db_source, ts
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                db_name,
                int(tokens),
                float(wall_ms),
                float(store_ms),
                patch_id,
                db_source,
                ts,
            ),
        )
        conn.commit()
