import logging
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

