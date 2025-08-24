import json
import importlib
import threading


def fresh_db_router():
    import db_router
    return importlib.reload(db_router)


def test_get_connection_routing(tmp_path):
    db_router = fresh_db_router()
    router = db_router.DBRouter("test", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))

    # Create tables in respective databases
    router.shared_conn.execute("CREATE TABLE bots(id INTEGER, source_menace_id TEXT NOT NULL)")
    router.local_conn.execute("CREATE TABLE models(id INTEGER)")
    router.shared_conn.commit()
    router.local_conn.commit()

    shared_conn = router.get_connection("bots", "write")
    shared_conn.execute("INSERT INTO bots(id, source_menace_id) VALUES (1, ?)", (router.menace_id,))
    shared_conn.commit()
    local_conn = router.get_connection("models", "write")
    local_conn.execute("INSERT INTO models(id) VALUES (1)")
    local_conn.commit()

    assert router.shared_conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1
    assert router.local_conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 1


def test_get_connection_unknown_and_denied(tmp_path):
    db_router = fresh_db_router()
    router = db_router.DBRouter("test", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))

    import pytest

    with pytest.raises(ValueError):
        router.get_connection("unknown_table")
    with pytest.raises(ValueError):
        router.get_connection("capital_ledger")


def test_get_connection_thread_safety(tmp_path):
    db_router = fresh_db_router()
    router = db_router.DBRouter("test", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    router.shared_conn.isolation_level = None
    router.local_conn.isolation_level = None
    router.shared_conn.execute("CREATE TABLE bots(id INTEGER, source_menace_id TEXT NOT NULL)")
    router.local_conn.execute("CREATE TABLE models(id INTEGER)")

    def worker(table, value):
        conn = router.get_connection(table, "write")
        cur = conn.cursor()
        if table == "bots":
            cur.execute(
                "INSERT INTO bots(id, source_menace_id) VALUES (?, ?)",
                (value, router.menace_id),
            )
            cur.execute(
                "SELECT id FROM bots WHERE id=? AND source_menace_id=?",
                (value, router.menace_id),
            )
        else:
            cur.execute(f"INSERT INTO {table}(id) VALUES (?)", (value,))
            cur.execute(f"SELECT id FROM {table} WHERE id=?", (value,))
        assert cur.fetchone()[0] == value
        cur.close()

    threads = []
    for i in range(5):
        t1 = threading.Thread(target=worker, args=("bots", i))
        t2 = threading.Thread(target=worker, args=("models", i))
        threads.extend([t1, t2])
        t1.start()
        t2.start()

    for t in threads:
        t.join()

    assert router.shared_conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 5
    assert router.local_conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 5


def test_audit_log_written(tmp_path, monkeypatch):
    audit_file = tmp_path / "audit.log"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(audit_file))
    db_router = fresh_db_router()
    router = db_router.DBRouter("test", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    router.get_connection("bots")
    with open(audit_file, "r", encoding="utf-8") as fh:
        entry = json.loads(fh.readline())
    assert entry["table_name"] == "bots"
    assert entry["operation"] == "read"
    monkeypatch.delenv("DB_ROUTER_AUDIT_LOG", raising=False)
    fresh_db_router()
