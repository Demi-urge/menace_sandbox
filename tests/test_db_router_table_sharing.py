import importlib
import json
import sqlite3
import threading

import db_router
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


def test_recent_local_tables_route_to_local(tmp_path):
    """Ensure newly added local tables use the menace-specific database."""
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("one", str(local_db), str(shared_db))
    try:
        for table in (
            "sandbox_metrics",
            "roi_logs",
            "menace_config",
            "vector_metrics",
            "roi_telemetry",
            "roi_prediction_events",
            "results",
            "resolutions",
            "deployments",
            "bot_trials",
            "update_history",
            "roi_events",
        ):
            with router.get_connection(table) as conn:
                db_path = conn.execute("PRAGMA database_list").fetchall()[0][2]
                assert db_path == str(local_db)
    finally:
        router.close()


def test_shared_telemetry_routes_to_shared(tmp_path):
    """Verify the telemetry table is stored in the shared database."""
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    router = DBRouter("one", str(local_db), str(shared_db))
    try:
        with router.get_connection("telemetry") as conn:
            db_path = conn.execute("PRAGMA database_list").fetchall()[0][2]
            assert db_path == str(shared_db)
    finally:
        router.close()


def test_threaded_shared_and_local_with_audit(tmp_path, monkeypatch):
    """Concurrent read/write operations log table and menace details."""

    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    audit_log = tmp_path / "audit.log"
    monkeypatch.setenv("DB_ROUTER_AUDIT_LOG", str(audit_log))
    importlib.reload(db_router)

    router = db_router.DBRouter("alpha", str(local_db), str(shared_db))
    try:
        # Prepare shared and local tables
        with router.get_connection("bots", operation="write") as conn:
            conn.execute("CREATE TABLE bots (id INTEGER PRIMARY KEY, name TEXT)")
            conn.commit()
        with router.get_connection("models", operation="write") as conn:
            conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT)")
            conn.commit()

        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def write_shared(idx: int) -> None:
            try:
                barrier.wait()
                with router.get_connection("bots", operation="write") as conn:
                    conn.execute("INSERT INTO bots (name) VALUES (?)", (f"bot{idx}",))
                    conn.commit()
            except Exception as exc:  # pragma: no cover - capturing unexpected errors
                errors.append(exc)

        def write_local(idx: int) -> None:
            try:
                barrier.wait()
                with router.get_connection("models", operation="write") as conn:
                    conn.execute("INSERT INTO models (name) VALUES (?)", (f"model{idx}",))
                    conn.commit()
            except Exception as exc:  # pragma: no cover - capturing unexpected errors
                errors.append(exc)

        threads = [threading.Thread(target=write_shared, args=(i,)) for i in range(2)] + [
            threading.Thread(target=write_local, args=(i,)) for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

        # Verify final table contents
        with router.get_connection("bots") as conn:
            shared_rows = [row[0] for row in conn.execute("SELECT name FROM bots ORDER BY id")]
        with router.get_connection("models") as conn:
            local_rows = [row[0] for row in conn.execute("SELECT name FROM models ORDER BY id")]
        assert set(shared_rows) == {"bot0", "bot1"}
        assert set(local_rows) == {"model0", "model1"}
    finally:
        router.close()
        importlib.reload(db_router)

    entries = [json.loads(line) for line in audit_log.read_text().strip().splitlines()]
    assert {e["table_name"] for e in entries} >= {"bots", "models"}
    assert {e["operation"] for e in entries} >= {"write", "read"}
    assert {e["menace_id"] for e in entries} == {"alpha"}
