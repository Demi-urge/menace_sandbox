import sqlite3
from db_router import DBRouter


def test_shared_table_persists_across_instances(tmp_path):
    shared_db = tmp_path / "shared.db"
    local1 = tmp_path / "local1.db"
    local2 = tmp_path / "local2.db"

    router1 = DBRouter("one", str(local1), str(shared_db))
    router2 = DBRouter("two", str(local2), str(shared_db))

    # Write to a shared table via router1
    with router1.get_connection("bots") as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bots (id INTEGER PRIMARY KEY, name TEXT)"
        )
        conn.execute("INSERT INTO bots (name) VALUES (?)", ("alpha",))
        conn.commit()

    # Ensure router2 can read the data from the shared table
    with router2.get_connection("bots") as conn:
        rows = list(conn.execute("SELECT name FROM bots"))
        assert rows == [("alpha",)]


def test_local_table_isolated_between_instances(tmp_path):
    shared_db = tmp_path / "shared.db"
    local1 = tmp_path / "local1.db"
    local2 = tmp_path / "local2.db"

    router1 = DBRouter("one", str(local1), str(shared_db))
    router2 = DBRouter("two", str(local2), str(shared_db))

    # Write to a local table via router1
    with router1.get_connection("models") as conn:
        conn.execute(
            "CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT)"
        )
        conn.execute("INSERT INTO models (name) VALUES (?)", ("alpha",))
        conn.commit()

    # Router2 should not see the table or data
    with router2.get_connection("models") as conn:
        tables = list(
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
            )
        )
        assert tables == []
