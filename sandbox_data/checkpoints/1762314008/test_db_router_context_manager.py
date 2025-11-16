import importlib
import sqlite3

import pytest

import db_router


def test_context_manager_closes_connections(tmp_path):
    """DBRouter context manager should close both database connections."""
    importlib.reload(db_router)
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    with db_router.DBRouter("ctx", str(local_db), str(shared_db)) as router:
        router.get_connection("bots")
        router.get_connection("models")
    with pytest.raises(sqlite3.ProgrammingError):
        router.local_conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        router.shared_conn.execute("SELECT 1")
