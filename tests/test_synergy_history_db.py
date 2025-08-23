import json
from pathlib import Path
from unittest.mock import patch

import db_router
import pytest
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


def test_connect_uses_router(tmp_path: Path) -> None:
    router = db_router.DBRouter("syn", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        with pytest.raises(AttributeError):
            shd.connect(tmp_path / "history.db", router=router)
        gc.assert_called_with("synergy_history")
    assert "synergy_history" in db_router.LOCAL_TABLES
    router.close()
