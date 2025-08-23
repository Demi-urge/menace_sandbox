import logging
import threading
import pytest

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


def test_get_connection_logs(tmp_path, caplog):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    test_logger = logging.getLogger("db_router_test")
    router = DBRouter(
        "test",
        str(local_db),
        str(shared_db),
        logger=test_logger,
        log_level=logging.INFO,
    )

    try:
        with caplog.at_level(logging.INFO, logger="db_router_test"):
            router.get_connection("bots")
        assert "table 'bots'" in caplog.text
        assert "shared connection" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="db_router_test"):
            router.get_connection("models")
        assert "table 'models'" in caplog.text
        assert "local connection" in caplog.text
    finally:
        router.close()

    caplog.clear()
    router = DBRouter(
        "test",
        str(local_db),
        str(shared_db),
        logger=test_logger,
        log_level=None,
    )
    try:
        with caplog.at_level(logging.INFO, logger="db_router_test"):
            router.get_connection("bots")
        assert caplog.text == ""
    finally:
        router.close()


def test_get_connection_thread_safe(tmp_path):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("test", str(local_db), str(shared_db))

    results = []
    errors = []

    def worker(table_name):
        try:
            conn = router.get_connection(table_name)
            results.append((_db_path(conn), table_name))
        except Exception as exc:  # pragma: no cover - capturing unexpected errors
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(t,)) for t in ("bots", "models") * 10
    ]
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

