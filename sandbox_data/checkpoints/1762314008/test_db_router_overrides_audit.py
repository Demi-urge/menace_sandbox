import json
import importlib
import sys
import types

import pytest

import db_router


def _db_path(conn: "sqlite3.Connection") -> str:
    return conn.execute("PRAGMA database_list").fetchall()[0][2]


def test_env_and_config_overrides_audit_and_metrics(tmp_path, monkeypatch):
    cfg = {
        "shared": ["cfg_shared"],
        "local": ["cfg_local"],
        "deny": ["cfg_deny"],
        "audit_log": str(tmp_path / "audit.log"),
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    monkeypatch.setenv("DB_ROUTER_SHARED_TABLES", "env_shared")
    monkeypatch.setenv("DB_ROUTER_LOCAL_TABLES", "env_local")
    monkeypatch.setenv("DB_ROUTER_DENY_TABLES", "env_deny")
    monkeypatch.setenv("DB_ROUTER_CONFIG", str(cfg_path))

    importlib.reload(db_router)

    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = db_router.DBRouter("alpha", str(local_db), str(shared_db))

    calls: list[tuple[str, str, str, int]] = []
    dummy = types.SimpleNamespace(
        record_table_access=lambda menace, table, kind, count: calls.append(
            (menace, table, kind, count)
        )
    )
    monkeypatch.setitem(sys.modules, "telemetry_backend", dummy)

    try:
        with router.get_connection("env_shared", operation="write") as conn:
            assert _db_path(conn) == str(shared_db)
        with router.get_connection("cfg_shared") as conn:
            assert _db_path(conn) == str(shared_db)
        with router.get_connection("env_local") as conn:
            assert _db_path(conn) == str(local_db)
        with router.get_connection("cfg_local") as conn:
            assert _db_path(conn) == str(local_db)

        with pytest.raises(ValueError):
            router.get_connection("env_deny")
        with pytest.raises(ValueError):
            router.get_connection("cfg_deny")

        counts = router.get_access_counts(flush=True)
    finally:
        router.close()
        monkeypatch.delenv("DB_ROUTER_SHARED_TABLES", raising=False)
        monkeypatch.delenv("DB_ROUTER_LOCAL_TABLES", raising=False)
        monkeypatch.delenv("DB_ROUTER_DENY_TABLES", raising=False)
        monkeypatch.delenv("DB_ROUTER_CONFIG", raising=False)
        importlib.reload(db_router)

    assert counts["shared"]["env_shared"] == 1
    assert counts["shared"]["cfg_shared"] == 1
    assert counts["local"]["env_local"] == 1
    assert counts["local"]["cfg_local"] == 1

    assert sorted(calls) == sorted(
        [
            ("alpha", "env_shared", "shared", 1),
            ("alpha", "cfg_shared", "shared", 1),
            ("alpha", "env_local", "local", 1),
            ("alpha", "cfg_local", "local", 1),
        ]
    )

    audit_log = tmp_path / "audit.log"
    entries = [json.loads(line) for line in audit_log.read_text().splitlines()]
    tables = {e["table_name"] for e in entries}
    assert tables == {"env_shared", "cfg_shared", "env_local", "cfg_local"}
    assert {e["menace_id"] for e in entries} == {"alpha"}

def test_runtime_deny_tables(tmp_path, monkeypatch):
    importlib.reload(db_router)
    monkeypatch.setattr(db_router, "DENY_TABLES", {"bots"})
    router = db_router.DBRouter("beta", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    try:
        with pytest.raises(ValueError):
            router.get_connection("bots")
    finally:
        router.close()
        importlib.reload(db_router)
