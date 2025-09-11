from pathlib import Path
import sys

from db_router import GLOBAL_ROUTER, init_db_router
from dynamic_path_router import resolve_path

router = GLOBAL_ROUTER or init_db_router(
    "cleanup_visual_agent_code",
    local_db_path=str(resolve_path("menace.db")),
    shared_db_path=str(resolve_path("code.db")),
)


def upgrade(path: str | Path | None = None) -> None:
    """Remove rows referencing visual_agent from the code table."""
    conn = router.get_connection("code")
    try:
        conn.execute(
            "DELETE FROM code WHERE code LIKE '%visual_agent%' OR summary LIKE '%visual_agent%'"
        )
        conn.commit()
    finally:
        pass


if __name__ == "__main__":
    upgrade(sys.argv[1] if len(sys.argv) > 1 else None)
