import logging
import threading

import pytest

import importlib
import json
import sys
import types

import db_router
from db_router import DBRouter


def _db_path(conn):
    return conn.execute("PRAGMA database_list").fetchall()[0][2]


def test_get_connection_routes_tables(tmp_path):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    try:
        for table in ("bots", "errors"):
            with router.get_connection(table) as conn:
                assert _db_path(conn) == str(shared_db)

        with router.get_connection("models") as conn:
            assert _db_path(conn) == str(local_db)
    finally:
        router.close()


def test_unknown_table_raises(tmp_path):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    try:
        with pytest.raises(ValueError):
            router.get_connection("unknown")
    finally:
        router.close()


def test_structured_logging_for_shared_table(tmp_path, caplog):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    try:
        with caplog.at_level(logging.INFO):
            router.get_connection("bots")
            router.get_connection("models")
        records = [json.loads(r.msg) for r in caplog.records]
        assert len(records) == 1
        entry = records[0]
        assert entry["menace_id"] == "test"
        assert entry["table_name"] == "bots"
        assert entry["operation"] == "read"
        assert "timestamp" in entry
    finally:
        router.close()


def test_local_table_logging_suppressed(tmp_path, caplog):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    try:
        with caplog.at_level(logging.INFO):
            router.get_connection("models")
        assert caplog.records == []
    finally:
        router.close()


def test_table_metrics_callback(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    calls: list[tuple[str, str, str, int]] = []
    dummy = types.SimpleNamespace(
        record_table_access=lambda menace, table, op, count=1: calls.append(
            (menace, table, op, count)
        )
    )
    monkeypatch.setitem(sys.modules, "telemetry_backend", dummy)
    try:
        router.get_connection("bots")
        router.get_connection("models")
        router.get_access_counts(flush=True)
    finally:
        router.close()
    assert calls == [
        ("test", "bots", "shared", 1),
        ("test", "models", "local", 1),
    ]


def test_get_connection_thread_safe(tmp_path):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    results: list[tuple[str, str]] = []
    errors: list[Exception] = []
    tables = ("bots", "models") * 10
    barrier = threading.Barrier(len(tables))

    def worker(table_name: str) -> None:
        try:
            barrier.wait()
            with router.get_connection(table_name) as conn:
                results.append((_db_path(conn), table_name))
        except Exception as exc:  # pragma: no cover - capturing unexpected errors
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in tables]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    shared_paths = [p for p, table in results if table == "bots"]
    local_paths = [p for p, table in results if table == "models"]
    assert all(path == str(shared_db) for path in shared_paths)
    assert all(path == str(local_db) for path in local_paths)

    router.close()


def test_access_metrics(tmp_path):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    try:
        router.get_connection("bots")
        router.get_connection("bots")
        router.get_connection("models")
        counts = router.get_access_counts()
        assert counts["shared"]["bots"] == 2
        assert counts["local"]["models"] == 1
    finally:
        router.close()


def test_table_lists_from_env(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    monkeypatch.setenv("DB_ROUTER_SHARED_TABLES", "env_shared")
    monkeypatch.setenv("DB_ROUTER_LOCAL_TABLES", "env_local")
    monkeypatch.setenv("DB_ROUTER_DENY_TABLES", "bots")
    importlib.reload(db_router)
    router = db_router.DBRouter("test", str(local_db), str(shared_db))
    try:
        with router.get_connection("env_shared") as conn:
            assert _db_path(conn) == str(shared_db)
        with router.get_connection("env_local") as conn:
            assert _db_path(conn) == str(local_db)
        with pytest.raises(ValueError):
            router.get_connection("bots")
    finally:
        router.close()
    monkeypatch.delenv("DB_ROUTER_SHARED_TABLES", raising=False)
    monkeypatch.delenv("DB_ROUTER_LOCAL_TABLES", raising=False)
    monkeypatch.delenv("DB_ROUTER_DENY_TABLES", raising=False)
    importlib.reload(db_router)


def test_table_lists_from_config(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    cfg = {"shared": ["cfg_shared"], "local": ["cfg_local"], "deny": ["bots"]}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("DB_ROUTER_CONFIG", str(cfg_path))
    importlib.reload(db_router)
    router = db_router.DBRouter("test", str(local_db), str(shared_db))
    try:
        with router.get_connection("cfg_shared") as conn:
            assert _db_path(conn) == str(shared_db)
        with router.get_connection("cfg_local") as conn:
            assert _db_path(conn) == str(local_db)
        with pytest.raises(ValueError):
            router.get_connection("bots")
    finally:
        router.close()
    monkeypatch.delenv("DB_ROUTER_CONFIG", raising=False)
    importlib.reload(db_router)


def test_logging_respects_env_level(tmp_path, monkeypatch, caplog):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    monkeypatch.setenv("DB_ROUTER_LOG_LEVEL", "WARNING")
    importlib.reload(db_router)
    router = db_router.DBRouter("test", str(local_db), str(shared_db))
    try:
        with caplog.at_level(logging.INFO):
            router.get_connection("bots")
        assert caplog.records == []
    finally:
        router.close()
    monkeypatch.delenv("DB_ROUTER_LOG_LEVEL", raising=False)
    importlib.reload(db_router)

