import sqlite3
from pathlib import Path
import sys


def upgrade(path: str | Path) -> None:
    """Add efficiency and bottleneck columns if missing."""
    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(evolution_history)").fetchall()]
    if "efficiency" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN efficiency REAL DEFAULT 0")
    if "bottleneck" not in cols:
        conn.execute("ALTER TABLE evolution_history ADD COLUMN bottleneck REAL DEFAULT 0")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: upgrade_evolution_history.py <db_path>")
    else:
        upgrade(sys.argv[1])
