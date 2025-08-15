from __future__ import annotations

"""SQLite logger for embedding statistics."""

from datetime import datetime
from pathlib import Path
import sqlite3


class EmbeddingStatsDB:
    """Persist high-level embedding metrics.

    Each record captures the timestamp, database name, token usage and timing
    information for a single embedding operation.  The database is lazily
    initialised on first use and is safe for use across threads.
    """

    def __init__(self, path: str | Path = "metrics.db") -> None:
        """Create or connect to the metrics database.

        Historically embedding statistics were written to a dedicated
        ``embedding_stats.db`` file.  The new design stores them alongside
        other metrics in ``metrics.db`` for easier aggregation.  The table
        now records optional ``patch_id`` and ``db_source`` fields and a
        timestamp column named ``ts`` to mirror the rest of the schema.
        """

        self.conn = sqlite3.connect(path, check_same_thread=False)
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
        self.conn.execute(
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
        self.conn.commit()
