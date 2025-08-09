import sqlite3
from pathlib import Path
import sys


def upgrade(path: str | Path) -> None:
    """Upgrade evolution_history schema with new columns if missing."""
    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(evolution_history)").fetchall()]
    if "efficiency" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN efficiency REAL DEFAULT 0")
    if "bottleneck" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN bottleneck REAL DEFAULT 0")
    if "predicted_roi" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN predicted_roi REAL DEFAULT 0")
    if "patch_id" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN patch_id INTEGER")
    if "workflow_id" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN workflow_id INTEGER")
    if "trending_topic" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN trending_topic TEXT")
    if "reason" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN reason TEXT")
    if "trigger" not in cols:
        conn.execute('ALTER TABLE evolution_history ADD COLUMN "trigger" TEXT')
    if "performance" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN performance REAL DEFAULT 0")
    if "parent_id" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN parent_id INTEGER")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: upgrade_evolution_history.py <db_path>")
    else:
        upgrade(sys.argv[1])
