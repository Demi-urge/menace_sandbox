import json
import importlib


def test_db_access_audit(tmp_path, monkeypatch):
    log_path = tmp_path / "shared_db_access.log"
    local_db = tmp_path / "local.db"
    shared_db = tmp_path / "shared.db"

    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(log_path))
    import db_router
    importlib.reload(db_router)

    router = db_router.init_db_router("alpha", str(local_db), str(shared_db))
    try:
        conn = router.get_connection("telemetry")
        cur = conn.cursor()
        cur.execute("CREATE TABLE telemetry (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()
        log_path.write_text("")
        cur.execute("SELECT * FROM telemetry")
        cur.execute("INSERT INTO telemetry (data) VALUES (?)", ("foo",))
        conn.commit()
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        entries = [e for e in entries if "action" in e]
        assert len(entries) == 2
        read, write = entries
        assert read["action"] == "read"
        assert read["table"] == "telemetry"
        assert read["rows"] == 0
        assert read["menace_id"] == "alpha"
        assert write["action"] == "write"
        assert write["table"] == "telemetry"
        assert write["rows"] == 1
        assert write["menace_id"] == "alpha"
    finally:
        router.close()
