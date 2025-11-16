import json
import importlib

import db_router


def test_accesses_are_audited(tmp_path, monkeypatch):
    audit_log = tmp_path / "audit.log"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(audit_log))
    importlib.reload(db_router)

    router = db_router.DBRouter("alpha", str(tmp_path), str(tmp_path / "shared.db"))
    try:
        with router.get_connection("bots", operation="write") as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS bots (id INTEGER)")
            conn.commit()
        with router.get_connection("models") as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS models (id INTEGER)")
            conn.commit()
    finally:
        router.close()
        importlib.reload(db_router)

    entries = [json.loads(line) for line in audit_log.read_text().strip().splitlines()]
    entries = [e for e in entries if "table_name" in e]
    assert {e["menace_id"] for e in entries} == {"alpha"}
    assert {e["table_name"] for e in entries} == {"bots", "models"}
    assert {e["operation"] for e in entries} == {"write", "read"}
    for e in entries:
        assert "timestamp" in e


def test_shared_table_access_logged(tmp_path, monkeypatch):
    audit_log = tmp_path / "audit.log"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(audit_log))
    importlib.reload(db_router)

    router = db_router.DBRouter("alpha", str(tmp_path), str(tmp_path / "shared.db"))
    try:
        with router.get_connection("bots", operation="write") as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS bots (id INTEGER)")
            conn.commit()
    finally:
        router.close()
        importlib.reload(db_router)

    entries = [json.loads(line) for line in audit_log.read_text().splitlines()]
    entries = [e for e in entries if "table_name" in e]
    assert entries[0]["table_name"] == "bots"
    assert entries[0]["operation"] == "write"
