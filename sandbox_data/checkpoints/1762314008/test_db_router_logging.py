import json
import importlib
import sqlite3

def test_db_router_audit_log(tmp_path, monkeypatch):
    log_path = tmp_path / "shared_db_access.log"
    local_db = tmp_path / "local.db"
    shared_db = tmp_path / "shared.db"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(log_path))
    import db_router
    importlib.reload(db_router)
    with sqlite3.connect(shared_db) as pre:
        pre.execute("CREATE TABLE telemetry (id INTEGER PRIMARY KEY, data TEXT)")
    router = db_router.DBRouter("alpha", str(local_db), str(shared_db))
    try:
        conn = router.get_connection("telemetry")
        cur = conn.cursor()
        cur.execute("SELECT * FROM telemetry")
        cur.execute("INSERT INTO telemetry (data) VALUES (?)", ("foo",))
        conn.commit()
        with log_path.open() as fh:
            entries = [json.loads(line) for line in fh]
        entries = [e for e in entries if "action" in e]
        assert len(entries) == 2
        read, write = entries
        assert read["action"] == "read"
        assert read["table"] == "telemetry"
        assert read["rows"] == 0
        assert read["menace_id"] == "alpha"
        assert isinstance(read.get("timestamp"), str) and read["timestamp"]
        assert write["action"] == "write"
        assert write["table"] == "telemetry"
        assert write["rows"] == 1
        assert write["menace_id"] == "alpha"
        assert isinstance(write.get("timestamp"), str) and write["timestamp"]
    finally:
        router.close()
        for path in (log_path, local_db, shared_db):
            if path.exists():
                path.unlink()
