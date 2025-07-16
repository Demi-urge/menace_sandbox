from pathlib import Path
import types
import logging

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import Column, Integer, String, MetaData, Table, create_engine, select

import menace.central_database_bot as cdb


def setup_db(path: Path):
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    meta = MetaData()
    Table(
        "items",
        meta,
        Column("id", Integer, primary_key=True),
        Column("name", String),
    )
    meta.create_all(engine)
    return url


def test_process_queue(tmp_path: Path):
    db_url = setup_db(tmp_path / "db.sqlite")
    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    bot = cdb.CentralDatabaseBot(db_url=db_url, db_router=router)
    p1 = cdb.Proposal(
        operation="insert",
        target_table="items",
        payload={"name": "a"},
        origin_bot_id="test",
    )
    p2 = cdb.Proposal(
        operation="insert",
        target_table="items",
        payload={"name": "b"},
        origin_bot_id="test",
    )
    bot.enqueue(p1)
    bot.enqueue(p2)
    bot.process_all()
    tbl = bot.meta.tables["items"]
    with bot.engine.begin() as conn:
        rows = conn.execute(select(tbl).order_by(tbl.c.id)).fetchall()
    assert [r[1] for r in rows] == ["a", "b"]
    assert bot.results and bot.results[0].status == "committed"
    assert "items" in router.terms


def test_invalid_proposal(tmp_path: Path):
    db_url = setup_db(tmp_path / "db.sqlite")
    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    bot = cdb.CentralDatabaseBot(db_url=db_url, db_router=router)
    bad = cdb.Proposal(
        operation="insert",
        target_table="missing",
        payload={"name": "x"},
        origin_bot_id="test",
    )
    bot.enqueue(bad)
    bot.process_once()
    with bot.engine.begin() as conn:
        rows = conn.execute(select(bot.invalid_table)).fetchall()
    assert rows
    assert bot.results and bot.results[0].status == "invalid"
    assert "missing" in router.terms


def test_redis_failure_falls_back(tmp_path: Path, caplog):
    db_url = setup_db(tmp_path / "db.sqlite")
    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    bot = cdb.CentralDatabaseBot(db_url=db_url, db_router=router)
    class FakeRedis:
        def xadd(self, *a, **k):
            raise RuntimeError("boom")
    bot.client = FakeRedis()
    caplog.set_level(logging.WARNING)
    p = cdb.Proposal(
        operation="insert",
        target_table="items",
        payload={"name": "x"},
        origin_bot_id="test",
    )
    bot.enqueue(p)
    assert bot.queue and bot.queue[0] == p
    assert "enqueue to redis failed" in caplog.text
    assert "falling back to local queue" in caplog.text
