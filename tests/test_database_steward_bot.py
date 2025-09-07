from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

import menace.database_steward_bot as dsb
import menace.error_bot as eb
from sqlalchemy import Column, Integer, String, Table


class DummyBuilder:
    def refresh_db_weights(self):
        pass


def test_version_and_lock(tmp_path: Path):
    repo = tmp_path / "repo"
    bot = dsb.DatabaseStewardBot(sql_url=f"sqlite:///{tmp_path / 'db.sqlite'}", repo_path=repo)
    file = repo / "a.txt"
    file.write_text("hi")
    commit = bot.version_file(file)
    assert commit
    assert bot.lock("a", "t") is True
    assert bot.lock("a", "x") is False
    bot.unlock("a", "t")
    assert bot.lock("a", "x") is True


def test_audit_and_dedup(tmp_path: Path):
    bot = dsb.DatabaseStewardBot(sql_url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    bot.sql.add("temp")
    bot.sql.add("temp")
    assert bot.audit() == []
    bot.deduplicate()
    rows = bot.sql.fetch()
    assert len(rows) == 1


def test_error_bot_fix(tmp_path: Path):
    admin = tmp_path / "admin.py"  # path-ignore
    admin.write_text("print('x')\n")
    err = eb.ErrorDB(tmp_path / "e.db")
    ebot = eb.ErrorBot(err, context_builder=DummyBuilder())
    bot = dsb.DatabaseStewardBot(
        sql_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        error_bot=ebot,
    )
    with bot.sql.engine.begin() as conn:
        bot.sql.templates.drop(conn)
    bot.sql.meta.remove(bot.sql.templates)
    bot.sql.templates = Table(
        "templates",
        bot.sql.meta,
        Column("id", Integer, primary_key=True),
        Column("name", String),
        extend_existing=True,
    )
    bot.sql.meta.create_all(bot.sql.engine)
    issues = bot.resolve_management_issues(admin)
    assert "schema_drift" in issues
    assert "expected_schema" in admin.read_text()
