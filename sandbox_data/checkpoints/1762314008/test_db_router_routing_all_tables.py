import importlib
import json

import db_router


def test_routing_for_all_tables(tmp_path, monkeypatch):
    """Every declared table routes to the expected database."""
    importlib.reload(db_router)
    audit_log = tmp_path / "audit.log"
    monkeypatch.setattr(db_router, "_audit_log_path", str(audit_log))
    monkeypatch.setattr(db_router, "DENY_TABLES", set())
    router = db_router.DBRouter("t1", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    try:
        for table in db_router.SHARED_TABLES:
            assert router.get_connection(table) is router.shared_conn
        for table in db_router.LOCAL_TABLES:
            assert router.get_connection(table) is router.local_conn
        counts = router.get_access_counts()
        assert counts["shared"] == {t: 1 for t in db_router.SHARED_TABLES}
        assert counts["local"] == {t: 1 for t in db_router.LOCAL_TABLES}
        lines = audit_log.read_text().strip().splitlines()
        assert len(lines) == len(db_router.SHARED_TABLES) + len(db_router.LOCAL_TABLES)
        first = json.loads(lines[0])
        assert first["menace_id"] == "t1"
    finally:
        router.close()
