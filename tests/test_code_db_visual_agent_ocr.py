import sqlite3
import shutil
import tempfile
from dynamic_path_router import resolve_path
from scripts.purge_visual_agent_ocr import purge


def test_no_visual_agent_entries():
    db_path = resolve_path("code.db")
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        shutil.copy(db_path, tmp.name)
        purge(tmp.name)
        conn = sqlite3.connect(tmp.name)  # noqa: SQL001
        cur = conn.cursor()
        count = cur.execute(
            (
                "SELECT COUNT(*) FROM code WHERE code LIKE '%visual_agent%'"
                " OR summary LIKE '%visual_agent%'"
            )
        ).fetchone()[0]
        conn.close()
        assert count == 0, f"Found {count} forbidden visual agent entries"
