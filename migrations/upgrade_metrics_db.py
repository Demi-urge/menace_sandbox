import sqlite3
from pathlib import Path
import sys


def upgrade(path: str | Path) -> None:
    """Upgrade metrics.db schema with vector and patch metrics."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS patch_outcomes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patch_id TEXT,
            success INTEGER,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_outcomes_ts ON patch_outcomes(ts)"
        )
        # ensure embedding_metrics table has required columns
        cols = [r[1] for r in conn.execute("PRAGMA table_info(embedding_metrics)").fetchall()]
        if "index_latency" not in cols:
            conn.execute(
                "ALTER TABLE embedding_metrics ADD COLUMN index_latency REAL"
            )
        if "tokens" not in cols:
            conn.execute(
                "ALTER TABLE embedding_metrics ADD COLUMN tokens INTEGER"
            )
        if "wall_time" not in cols:
            conn.execute(
                "ALTER TABLE embedding_metrics ADD COLUMN wall_time REAL"
            )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(retrieval_metrics)").fetchall()]
        if not cols:
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS retrieval_metrics(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin_db TEXT,
                record_id TEXT,
                rank INTEGER,
                hit INTEGER,
                tokens INTEGER,
                score REAL,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: upgrade_metrics_db.py <db_path>")
    else:
        upgrade(sys.argv[1])
