import argparse
import sqlite3
from dynamic_path_router import resolve_path


def purge(db_path: str) -> None:
    conn = sqlite3.connect(db_path)  # noqa: SQL001
    cur = conn.cursor()
    cur.execute(
        (
            "DELETE FROM code WHERE code LIKE '%visual_agent%'"
            " OR summary LIKE '%visual_agent%'"
        )
    )
    conn.commit()
    conn.execute("VACUUM")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Purge visual agent hooks from a code database",
    )
    parser.add_argument(
        "db",
        nargs="?",
        default=str(resolve_path("code.db")),
        help="path to SQLite code database",
    )
    args = parser.parse_args()
    purge(args.db)
