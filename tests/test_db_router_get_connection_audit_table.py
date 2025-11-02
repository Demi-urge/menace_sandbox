import importlib

import audit


def test_get_connection_logs_to_audit_table(tmp_path, monkeypatch):
    log_path = tmp_path / "shared_db_access.log"
    importlib.reload(audit)
    monkeypatch.setattr(audit, "DEFAULT_LOG_PATH", log_path)
    monkeypatch.setenv("DB_ROUTER_AUDIT_TO_DB", "1")
    import db_router
    importlib.reload(db_router)

    local_db = tmp_path / "local.db"
    shared_db = tmp_path / "shared.db"
    router = db_router.DBRouter("alpha", str(local_db), str(shared_db))
    try:
        router.shared_conn.execute(
            "CREATE TABLE telemetry (id INTEGER PRIMARY KEY, data TEXT)"
        )
        router.shared_conn.commit()

        conn = router.get_connection("telemetry")
        cur = conn.cursor()
        cur.execute("SELECT * FROM telemetry")
        cur.execute("INSERT INTO telemetry (data) VALUES (?)", ("foo",))
        conn.commit()

        cur = router.shared_conn.cursor()
        cur.execute(
            'SELECT action, "table", rows, menace_id FROM shared_db_audit WHERE "table"=? ORDER BY ROWID',
            ("telemetry",),
        )
        entries = cur.fetchall()
        assert entries == [
            ("read", "telemetry", 0, "alpha"),
            ("write", "telemetry", 1, "alpha"),
        ]
    finally:
        router.close()
