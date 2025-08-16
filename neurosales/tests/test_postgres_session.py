import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from neurosales.sql_db import create_session


def test_postgres_session_creation():
    pytest.importorskip("psycopg2")
    # engine creation should not raise even if DB is unreachable
    Session = create_session("postgresql://user:pass@localhost/db")
    assert callable(Session)

