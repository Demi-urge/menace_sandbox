import importlib

import db_router
import pytest


def test_unknown_and_denied_tables(tmp_path, monkeypatch):
    """Unknown and denied tables should raise ValueError and log nothing."""
    importlib.reload(db_router)
    audit_log = tmp_path / "audit.log"
    monkeypatch.setattr(db_router, "_audit_log_path", str(audit_log))
    router = db_router.DBRouter("t2", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    try:
        with pytest.raises(ValueError):
            router.get_connection("unknown_table")
        monkeypatch.setattr(db_router, "DENY_TABLES", {"bots"})
        with pytest.raises(ValueError):
            router.get_connection("bots")
        assert not audit_log.exists() or audit_log.read_text() == ""
        assert router.get_access_counts() == {"shared": {}, "local": {}}
    finally:
        router.close()
