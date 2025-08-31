from pathlib import Path
import sys

from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("upgrade_patch_suggestions")


def upgrade(path: str | Path | None = None) -> None:
    """Add score and rationale columns to suggestions table."""
    local_router = router
    if path is not None:
        local_router = init_db_router("upgrade_patch_suggestions", str(path), str(path))
    conn = local_router.get_connection("patches")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(suggestions)").fetchall()]
    if "score" not in cols:
        conn.execute("ALTER TABLE suggestions ADD COLUMN score REAL DEFAULT 0")
    if "rationale" not in cols:
        conn.execute("ALTER TABLE suggestions ADD COLUMN rationale TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_suggestions_score ON suggestions(score)"
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        upgrade()
    else:
        upgrade(sys.argv[1])
