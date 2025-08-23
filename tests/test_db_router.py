import logging
import threading

import pytest

import importlib
import json

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


def test_table_logging(tmp_path, caplog):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    try:
        with caplog.at_level(logging.INFO):
            router.get_connection("bots")
            router.get_connection("models")
        assert (
            "Shared table 'bots' accessed by menace 'test' on thread 'MainThread'"
            in caplog.text
        )
        assert (
            "Local table 'models' accessed by menace 'test' on thread 'MainThread'"
            in caplog.text
        )
    finally:
        router.close()


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
            router.get_connection("models")
        assert "table 'bots' accessed" not in caplog.text
        assert "table 'models' accessed" not in caplog.text
    finally:
        router.close()
    monkeypatch.delenv("DB_ROUTER_LOG_LEVEL", raising=False)
    importlib.reload(db_router)

