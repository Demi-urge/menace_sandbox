import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlalchemy as sa
import pytest

from neurosales.sql_db import ensure_schema, create_session, run_migrations


def test_ensure_schema_runs_migrations(tmp_path, monkeypatch):
    db_path = tmp_path / "schema.db"
    url = f"sqlite:///{db_path}"
    called = False

    real_run = run_migrations

    def fake_run(db_url, revision="head"):
        nonlocal called
        called = True
        real_run(db_url, revision)

    monkeypatch.setattr("neurosales.sql_db.run_migrations", fake_run)

    ensure_schema(url)

    assert called
    engine = sa.create_engine(url)
    inspector = sa.inspect(engine)
    assert "user_profiles" in inspector.get_table_names()
    engine.dispose()
