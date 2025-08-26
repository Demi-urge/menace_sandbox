import json
import importlib


def test_db_router_logs_read_and_write(tmp_path, monkeypatch):
    log_path = tmp_path / "shared_db_access.log"
    local_db = tmp_path / "local.db"
    shared_db = tmp_path / "shared.db"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(log_path))

    import db_router

    importlib.reload(db_router)

    router = db_router.DBRouter("alpha", str(local_db), str(shared_db))
    try:
        router.execute_and_log(
            "telemetry",
            "CREATE TABLE telemetry (id INTEGER PRIMARY KEY, data TEXT)",
        )
        log_path.write_text("")

        router.execute_and_log(
            "telemetry", "INSERT INTO telemetry (data) VALUES (?)", ("foo",)
        )
        rows = router.execute_and_log("telemetry", "SELECT * FROM telemetry")
        assert rows == [(1, "foo")]

        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        entries = [e for e in entries if "action" in e]
        assert len(entries) == 2
        write_entry = next(e for e in entries if e["action"] == "write")
        read_entry = next(e for e in entries if e["action"] == "read")

        assert write_entry["menace_id"] == "alpha"
        assert write_entry["table"] == "telemetry"
        assert write_entry["rows"] == 1

        assert read_entry["menace_id"] == "alpha"
        assert read_entry["table"] == "telemetry"
        assert read_entry["rows"] == 1
    finally:
        router.close()
