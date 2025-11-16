from pathlib import Path
import sys

from db_router import GLOBAL_ROUTER, init_db_router


router = GLOBAL_ROUTER or init_db_router("upgrade_prompt_db")


def upgrade(path: str | Path | None = None) -> None:
    """Upgrade prompts.db schema with token usage and cost columns."""
    conn = router.get_connection("prompts")
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(prompts)").fetchall()]
        if "backend" not in cols:
            conn.execute("ALTER TABLE prompts ADD COLUMN backend TEXT")
        if "input_tokens" not in cols:
            conn.execute("ALTER TABLE prompts ADD COLUMN input_tokens INTEGER")
        if "output_tokens" not in cols:
            conn.execute("ALTER TABLE prompts ADD COLUMN output_tokens INTEGER")
        if "cost" not in cols:
            conn.execute("ALTER TABLE prompts ADD COLUMN cost REAL")
        conn.commit()
    finally:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        upgrade()
    else:
        upgrade(sys.argv[1])
