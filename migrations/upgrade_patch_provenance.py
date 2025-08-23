from pathlib import Path
import sys

from db_router import init_db_router, GLOBAL_ROUTER


def upgrade(path: str | Path | None = None) -> None:
    """Upgrade patch provenance schema with alert columns."""
    conn = GLOBAL_ROUTER.get_connection("patch_provenance")
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
        pass


if __name__ == "__main__":
    init_db_router("upgrade_patch_provenance")
    if len(sys.argv) < 2:
        upgrade()
    else:
        upgrade(sys.argv[1])
