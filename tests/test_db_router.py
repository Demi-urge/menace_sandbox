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

