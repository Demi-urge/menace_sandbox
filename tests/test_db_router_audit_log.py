import json
import importlib

import db_router


def test_shared_access_is_audited(tmp_path, monkeypatch):
    audit_log = tmp_path / "audit.log"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(audit_log))
    importlib.reload(db_router)

    router = db_router.DBRouter("alpha", str(tmp_path), str(tmp_path / "shared.db"))
    try:
        with router.get_connection("bots") as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS bots (id INTEGER)")
            conn.commit()
    finally:
        router.close()

    lines = audit_log.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["menace_id"] == "alpha"
    assert entry["table_name"] == "bots"
    assert "timestamp" in entry
