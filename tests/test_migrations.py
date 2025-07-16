import os
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")
from alembic.config import Config
from alembic import command

import menace.menace as mn


def test_run_migrations(tmp_path):
    db_url = f"sqlite:///{tmp_path/'menace.db'}"
    os.environ["DATABASE_URL"] = db_url
    os.environ["MENACE_SKIP_CREATE"] = "1"
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")

    db = mn.MenaceDB(url=db_url)
    names = set(db.meta.tables.keys())
    assert {"models", "bots", "workflows"} <= names
