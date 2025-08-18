import sqlite3
from pathlib import Path
import sys


def upgrade(path: str | Path) -> None:
    """Upgrade patch provenance schema with alert columns."""
    conn = sqlite3.connect(path)
    try:
        for table in ("patch_provenance", "patch_ancestry"):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if "license" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN license TEXT")
            if "semantic_alerts" not in cols:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN semantic_alerts TEXT"
                )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: upgrade_patch_provenance.py <db_path>")
    else:
        upgrade(sys.argv[1])
