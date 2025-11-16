import sqlite3
import sys

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from menace.db_router import DBRouter, SHARED_TABLES, LOCAL_TABLES


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(table=st.sampled_from(sorted(SHARED_TABLES | LOCAL_TABLES)))
@pytest.mark.skipif("numpy" not in sys.modules and __import__("importlib").util.find_spec("numpy") is None,
                    reason="numpy not available")
def test_dbrouter_query_all_fuzz(tmp_path, table):
    """Fuzz DBRouter.get_connection with valid tables."""
    router = DBRouter("fuzz", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    conn = router.get_connection(table)
    assert isinstance(conn, sqlite3.Connection)


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(name=st.text(min_size=1, max_size=10))
@pytest.mark.skipif("numpy" not in sys.modules and __import__("importlib").util.find_spec("numpy") is None,
                    reason="numpy not available")
def test_dbrouter_execute_query_fuzz(tmp_path, name):
    """Fuzz simple INSERT/SELECT operations through DBRouter."""
    router = DBRouter("fuzz", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    conn = router.get_connection("bots")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS bots (name TEXT, source_menace_id TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO bots(name, source_menace_id) VALUES (?, ?)",
        (name, router.menace_id),
    )
    conn.commit()
    rows = conn.execute("SELECT name FROM bots WHERE name=?", (name,)).fetchall()
    router.close()
    assert rows and rows[0][0] == name
