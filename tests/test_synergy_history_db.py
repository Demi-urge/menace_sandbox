import json
import sqlite3
from pathlib import Path

from menace import synergy_history_db as shd


def test_migrate_and_fetch(tmp_path: Path) -> None:
    json_file = tmp_path / "history.json"
    data = [{"synergy_roi": 0.2}, {"synergy_roi": 0.3}]
    json_file.write_text(json.dumps(data))

    db_file = tmp_path / "history.db"
    shd.migrate_json_to_db(json_file, db_file)

    conn = shd.connect(db_file)
    try:
        all_rows = shd.fetch_all(conn)
        assert all_rows == data
        latest = shd.fetch_latest(conn)
        assert latest == data[-1]
    finally:
        conn.close()
