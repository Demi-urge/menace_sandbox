import importlib
import types
import sys
import pytest

import db_router

# Stub heavy dependencies used by code_database
sys.modules.setdefault("auto_link", types.SimpleNamespace(auto_link=lambda mapping: (lambda f: f)))
sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
sys.modules.setdefault(
    "retry_utils",
    types.SimpleNamespace(publish_with_retry=lambda *a, **k: None, with_retry=lambda func, **_: func()),
)
sys.modules.setdefault("alert_dispatcher", types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={}))

import code_database

def _db_path(conn):
    return conn.execute("PRAGMA database_list").fetchall()[0][2]


def test_get_connection_returns_expected_db(tmp_path):
    """get_connection should return the shared or local DB based on table."""
    importlib.reload(db_router)
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = db_router.DBRouter("test", str(local_db), str(shared_db))
    try:
        with router.get_connection("bots") as conn:
            assert _db_path(conn) == str(shared_db)
        with router.get_connection("models") as conn:
            assert _db_path(conn) == str(local_db)
    finally:
        router.close()


def test_unknown_table_raises_value_error(tmp_path):
    importlib.reload(db_router)
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = db_router.DBRouter("test", str(local_db), str(shared_db))
    try:
        with pytest.raises(ValueError):
            router.get_connection("unknown_table")
    finally:
        router.close()


def test_denied_table_raises_value_error(tmp_path, monkeypatch):
    importlib.reload(db_router)
    monkeypatch.setattr(db_router, "DENY_TABLES", {"bots"})
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = db_router.DBRouter("test", str(local_db), str(shared_db))
    try:
        with pytest.raises(ValueError):
            router.get_connection("bots")
    finally:
        router.close()


def test_code_db_routes_to_shared_db(tmp_path):
    importlib.reload(db_router)
    importlib.reload(code_database)
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = db_router.DBRouter("test", str(local_db), str(shared_db))
    try:
        cdb = code_database.CodeDB(path=shared_db, router=router)
        cdb.add(code_database.CodeRecord(code="print('hi')", summary="demo"))
        counts = router.get_access_counts()
        assert counts["shared"].get("code")
    finally:
        router.close()
