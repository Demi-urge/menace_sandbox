import json
import importlib
import os

import db_router


def test_shared_table_logging(tmp_path, monkeypatch):
    log_path = tmp_path / "shared_db_access.log"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(log_path))
    importlib.reload(db_router)

    original_makedirs = os.makedirs

    def safe_makedirs(path, exist_ok=False):
        if path:
            original_makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr(db_router.os, "makedirs", safe_makedirs)

    router = db_router.DBRouter("alpha", str(tmp_path), ":memory:")
    try:
        router.execute_and_log(
            "information",
            "CREATE TABLE information (id INTEGER PRIMARY KEY, data TEXT)",
        )
        log_path.write_text("")

        router.execute_and_log(
            "information", "INSERT INTO information (data) VALUES (?)", ("foo",)
        )
        router.execute_and_log("information", "SELECT * FROM information")

        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        entries = [e for e in entries if "action" in e]
        assert len(entries) == 2
        write_entry = next(e for e in entries if e["action"] == "write")
        read_entry = next(e for e in entries if e["action"] == "read")

        assert write_entry["menace_id"] == "alpha"
        assert write_entry["table"] == "information"
        assert write_entry["rows"] == 1

        assert read_entry["menace_id"] == "alpha"
        assert read_entry["table"] == "information"
        assert read_entry["rows"] == 1
    finally:
        router.close()
        importlib.reload(db_router)
