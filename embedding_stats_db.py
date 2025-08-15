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

    def __init__(self, path: str | Path = "embedding_stats.db") -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_stats(
                timestamp TEXT,
                db_name TEXT,
                tokens INTEGER,
                wall_ms REAL,
                store_ms REAL
            )
            """
        )
        self.conn.commit()

    def log(self, db_name: str, tokens: int, wall_ms: float, store_ms: float) -> None:
        """Insert a new embedding statistics row."""

        ts = datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO embedding_stats(timestamp, db_name, tokens, wall_ms, store_ms) VALUES(?,?,?,?,?)",
            (ts, db_name, int(tokens), float(wall_ms), float(store_ms)),
        )
        self.conn.commit()
