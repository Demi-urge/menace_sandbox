import threading
from db_router import DBRouter


def _db_path(conn):
    return conn.execute("PRAGMA database_list").fetchall()[0][2]


def test_get_connection_routes_tables(tmp_path):
    shared_db = tmp_path / "shared.db"
    router = DBRouter("test", str(tmp_path), str(shared_db))
    local_db = tmp_path / "menace_test_local.db"
    try:
        for table in ("bots", "errors"):
            with router.get_connection(table) as conn:
                assert _db_path(conn) == str(shared_db)
        with router.get_connection("models") as conn:
            assert _db_path(conn) == str(local_db)
    finally:
        router.close()
        for path in (local_db, shared_db):
            if path.exists():
                path.unlink()


def test_thread_safe_connections(tmp_path):
    shared_db = tmp_path / "shared.db"
    router = DBRouter("test", str(tmp_path), str(shared_db))
    local_db = tmp_path / "menace_test_local.db"

    def insert_shared(i: int) -> None:
        with router.get_connection("bots") as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS bots (id INTEGER PRIMARY KEY, name TEXT)"
            )
            conn.execute("INSERT INTO bots (name) VALUES (?)", (f"bot{i}",))
            conn.commit()

    def insert_local(i: int) -> None:
        with router.get_connection("models") as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS models (id INTEGER PRIMARY KEY, name TEXT)"
            )
            conn.execute("INSERT INTO models (name) VALUES (?)", (f"model{i}",))
            conn.commit()

    threads = [
        threading.Thread(target=insert_shared, args=(i,)) for i in range(5)
    ] + [
        threading.Thread(target=insert_local, args=(i,)) for i in range(5)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        with router.get_connection("bots") as conn:
            shared_rows = conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0]
        with router.get_connection("models") as conn:
            local_rows = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        assert shared_rows == 5
        assert local_rows == 5
    finally:
        router.close()
        for path in (local_db, shared_db):
            if path.exists():
                path.unlink()
