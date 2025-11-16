import importlib
import json
import threading
import time
import sqlite3

import pytest

import db_router
from db_router import DBRouter


def _db_path(conn):
    return conn.execute("PRAGMA database_list").fetchall()[0][2]


def test_threaded_access_shared_and_local(tmp_path):
    """Concurrent reads from shared and local tables via one router."""
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("alpha", str(local_db), str(shared_db))

    try:
        # Prepare tables with sample data
        with router.get_connection("bots", operation="write") as conn:
            conn.execute(
                "CREATE TABLE bots (id INTEGER PRIMARY KEY, name TEXT, source_menace_id TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO bots (name, source_menace_id) VALUES (?, ?)",
                [(f"bot{i}", router.menace_id) for i in range(5)],
            )
            conn.commit()
        with router.get_connection("models", operation="write") as conn:
            conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT)")
            conn.executemany(
                "INSERT INTO models (name) VALUES (?)",
                [(f"model{i}",) for i in range(5)],
            )
            conn.commit()

        errors: list[Exception] = []
        shared_counts: list[int] = []
        local_counts: list[int] = []
        barrier = threading.Barrier(10)

        def read_shared() -> None:
            try:
                barrier.wait()
                with router.get_connection("bots") as conn:
                    count = conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0]
                    shared_counts.append(count)
            except Exception as exc:  # pragma: no cover - capturing unexpected errors
                errors.append(exc)

        def read_local() -> None:
            try:
                barrier.wait()
                with router.get_connection("models") as conn:
                    count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
                    local_counts.append(count)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=read_shared) for _ in range(5)] + [
            threading.Thread(target=read_local) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert all(c == 5 for c in shared_counts)
        assert all(c == 5 for c in local_counts)

        counts = router.get_access_counts()
        assert counts["shared"]["bots"] >= 6  # includes setup reads
        assert counts["local"]["models"] >= 6
    finally:
        router.close()


def test_concurrent_writes_shared_and_local(tmp_path):
    """Simultaneous writes to shared and local tables are isolated."""
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("alpha", str(local_db), str(shared_db))

    try:
        with router.get_connection("bots", operation="write") as conn:
            conn.execute(
                "CREATE TABLE bots (id INTEGER PRIMARY KEY, name TEXT, source_menace_id TEXT NOT NULL)"
            )
            conn.commit()
        with router.get_connection("models", operation="write") as conn:
            conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT)")
            conn.commit()

        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def write_shared(idx: int) -> None:
            try:
                barrier.wait()
                success = False
                for _ in range(5):
                    try:
                        with router.get_connection("bots", operation="write") as conn:
                            conn.execute(
                                "INSERT INTO bots (name, source_menace_id) VALUES (?, ?)",
                                (f"bot{idx}", router.menace_id),
                            )
                            conn.commit()
                        success = True
                        break
                    except sqlite3.OperationalError:
                        time.sleep(0.01)
                if not success:
                    errors.append(RuntimeError("write_shared failed"))
            except Exception as exc:  # pragma: no cover - unexpected
                errors.append(exc)

        def write_local(idx: int) -> None:
            try:
                barrier.wait()
                success = False
                for _ in range(5):
                    try:
                        with router.get_connection("models", operation="write") as conn:
                            conn.execute("INSERT INTO models (name) VALUES (?)", (f"model{idx}",))
                            conn.commit()
                        success = True
                        break
                    except sqlite3.OperationalError:
                        time.sleep(0.01)
                if not success:
                    errors.append(RuntimeError("write_local failed"))
            except Exception as exc:  # pragma: no cover - unexpected
                errors.append(exc)

        threads = [threading.Thread(target=write_shared, args=(i,)) for i in range(2)] + [
            threading.Thread(target=write_local, args=(i,)) for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        with router.get_connection("bots") as conn:
            assert conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 2
        with router.get_connection("models") as conn:
            assert conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 2
    finally:
        router.close()


def test_threaded_isolation_between_menace_ids(tmp_path):
    """Concurrent reads across menace_ids keep local data isolated."""
    shared_db = tmp_path / "shared.db"
    router1 = DBRouter("one", str(tmp_path), str(shared_db))
    router2 = DBRouter("two", str(tmp_path), str(shared_db))

    try:
        # Populate shared and local tables
        with router1.get_connection("bots", operation="write") as conn:
            conn.execute(
                "CREATE TABLE bots (id INTEGER PRIMARY KEY, name TEXT, source_menace_id TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO bots (name, source_menace_id) VALUES (?, ?)",
                [(f"one_bot{i}", router1.menace_id) for i in range(5)],
            )
            conn.executemany(
                "INSERT INTO bots (name, source_menace_id) VALUES (?, ?)",
                [(f"two_bot{i}", router1.menace_id) for i in range(5)],
            )
            conn.commit()
        for idx in range(5):
            with router1.get_connection("models", operation="write") as conn:
                if idx == 0:
                    conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT)")
                conn.execute("INSERT INTO models (name) VALUES (?)", (f"one_model{idx}",))
                conn.commit()
            with router2.get_connection("models", operation="write") as conn:
                if idx == 0:
                    conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT)")
                conn.execute("INSERT INTO models (name) VALUES (?)", (f"two_model{idx}",))
                conn.commit()

        errors: list[Exception] = []
        results_one: list[list[str]] = []
        results_two: list[list[str]] = []
        shared_counts: list[int] = []
        barrier = threading.Barrier(4)

        def read_shared(router: DBRouter) -> None:
            try:
                barrier.wait()
                with router.get_connection("bots") as conn:
                    count = conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0]
                    shared_counts.append(count)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def read_one_local() -> None:
            try:
                barrier.wait()
                with router1.get_connection("models") as conn:
                    names = [row[0] for row in conn.execute("SELECT name FROM models")]
                    results_one.append(names)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def read_two_local() -> None:
            try:
                barrier.wait()
                with router2.get_connection("models") as conn:
                    names = [row[0] for row in conn.execute("SELECT name FROM models")]
                    results_two.append(names)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=read_one_local),
            threading.Thread(target=read_two_local),
            threading.Thread(target=read_shared, args=(router1,)),
            threading.Thread(target=read_shared, args=(router2,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert results_one[0] == [f"one_model{i}" for i in range(5)]
        assert results_two[0] == [f"two_model{i}" for i in range(5)]
        assert shared_counts == [10, 10]
    finally:
        router1.close()
        router2.close()


def test_config_overrides_env_and_json(tmp_path, monkeypatch):
    """Env vars and JSON config combine to override table routing."""
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    cfg = {"shared": ["cfg_shared"], "local": ["cfg_local"], "deny": ["bots"]}
    cfg_path = tmp_path / "router_cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("DB_ROUTER_SHARED_TABLES", "env_shared")
    monkeypatch.setenv("DB_ROUTER_LOCAL_TABLES", "env_local")
    monkeypatch.setenv("DB_ROUTER_CONFIG", str(cfg_path))

    importlib.reload(db_router)
    router = db_router.DBRouter("alpha", str(local_db), str(shared_db))
    try:
        with router.get_connection("env_shared") as conn:
            assert _db_path(conn) == str(shared_db)
        with router.get_connection("cfg_shared") as conn:
            assert _db_path(conn) == str(shared_db)
        with router.get_connection("env_local") as conn:
            assert _db_path(conn) == str(local_db)
        with router.get_connection("cfg_local") as conn:
            assert _db_path(conn) == str(local_db)
        with pytest.raises(ValueError):
            router.get_connection("bots")
    finally:
        router.close()
        # Clean up and restore defaults
        monkeypatch.delenv("DB_ROUTER_SHARED_TABLES", raising=False)
        monkeypatch.delenv("DB_ROUTER_LOCAL_TABLES", raising=False)
        monkeypatch.delenv("DB_ROUTER_CONFIG", raising=False)
        importlib.reload(db_router)
