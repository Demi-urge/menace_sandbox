import sqlite3
from db_router import DBRouter


def test_shared_table_persists_across_instances(tmp_path):
    shared_db = tmp_path / "shared.db"

    # Local databases are created automatically based on menace_id
    router1 = DBRouter("one", str(tmp_path), str(shared_db))
    router2 = DBRouter("two", str(tmp_path), str(shared_db))

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

    router1 = DBRouter("one", str(tmp_path), str(shared_db))
    router2 = DBRouter("two", str(tmp_path), str(shared_db))

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


def test_shared_and_local_visibility_across_instances(tmp_path):
    """Data in shared tables propagates while local tables stay private."""

    shared_db = tmp_path / "shared.db"

    router1 = DBRouter("one", str(tmp_path), str(shared_db))
    router2 = DBRouter("two", str(tmp_path), str(shared_db))

    # Populate a shared table via router1
    with router1.get_connection("bots") as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bots (id INTEGER PRIMARY KEY, name TEXT)"
        )
        conn.execute("INSERT INTO bots (name) VALUES (?)", ("alpha",))
        conn.commit()

    # Router2 should observe the shared data
    with router2.get_connection("bots") as conn:
        rows = list(conn.execute("SELECT name FROM bots"))
        assert rows == [("alpha",)]

    # Populate a local table via router1
    with router1.get_connection("models") as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS models (id INTEGER PRIMARY KEY, name TEXT)"
        )
        conn.execute("INSERT INTO models (name) VALUES (?)", ("beta",))
        conn.commit()

    # Router2 should not see the local table
    with router2.get_connection("models") as conn:
        tables = list(
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
            )
        )
        assert tables == []
