import importlib
import logging

import pytest
import sqlalchemy as sa

import db_dedup
import db_router


@pytest.fixture
def router(tmp_path):
    """Initialize a fresh DBRouter pointing at temporary files."""
    path = tmp_path / "db.sqlite"
    return db_router.init_db_router("test", str(path), str(path))


def test_insert_if_unique_sa(tmp_path, caplog):
    """Verify SQLAlchemy-based duplicate insertion prevention."""

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'dedup.db'}")
    meta = sa.MetaData()
    tbl = sa.Table(
        "items",
        meta,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text),
        sa.Column("content_hash", sa.String, unique=True),
    )
    meta.create_all(engine)

    values = {"name": "alpha"}
    logger = logging.getLogger(__name__)
    with caplog.at_level(logging.WARNING):
        pk1 = db_dedup.insert_if_unique(
            tbl,
            values,
            ["name"],
            "m1",
            engine=engine,
            logger=logger,
        )
        assert pk1 == 1
        pk2 = db_dedup.insert_if_unique(
            tbl,
            values,
            ["name"],
            "m1",
            engine=engine,
            logger=logger,
        )
        assert pk2 is None
    assert "Duplicate insert ignored for items" in caplog.text

    with engine.connect() as conn:
        count = conn.execute(sa.select(sa.func.count()).select_from(tbl)).scalar_one()
        assert count == 1


def test_botdb_dedup(tmp_path, caplog, monkeypatch):
    path = tmp_path / "db.sqlite"
    router = db_router.init_db_router("test", str(path), str(path))
    from menace import bot_database as bdb
    importlib.reload(bdb)
    bdb.router = router

    monkeypatch.setattr(bdb.BotDB, "_embed_record_on_write", lambda *a, **k: None)
    db = bdb.BotDB(path=tmp_path / "bots.db", vector_index_path=tmp_path / "bots.index")

    rec1 = bdb.BotRecord(name="alpha", purpose="p")
    id1 = db.add_bot(rec1)
    assert id1 > 0
    assert db.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        db.add_bot(bdb.BotRecord(name="alpha", purpose="p"))
    assert db.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1
    assert any("Duplicate insert ignored for bots" in r.message for r in caplog.records)

    db.add_bot(bdb.BotRecord(name="beta", purpose="p"))
    assert db.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 2


def test_enhancementdb_dedup(tmp_path, caplog, monkeypatch, router):
    from menace import chatgpt_enhancement_bot as ceb
    importlib.reload(ceb)

    monkeypatch.setattr(ceb.EnhancementDB, "add_embedding", lambda *a, **k: None)
    db = ceb.EnhancementDB(
        path=tmp_path / "enh.db",
        vector_index_path=tmp_path / "enh.index",
        metadata_path=tmp_path / "enh.meta",
        router=router,
    )

    enh1 = ceb.Enhancement(idea="idea1", rationale="r1")
    id1 = db.add(enh1)
    assert id1 > 0
    assert db.conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        db.add(ceb.Enhancement(idea="idea1", rationale="r1"))
    assert db.conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 1
    assert any("duplicate" in r.message.lower() for r in caplog.records)

    db.add(ceb.Enhancement(idea="idea2", rationale="r1"))
    assert db.conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 2


def test_errordb_dedup(tmp_path, caplog, monkeypatch, router):
    from menace import error_bot as eb
    importlib.reload(eb)

    monkeypatch.setattr(eb.ErrorDB, "add_embedding", lambda *a, **k: None)
    db = eb.ErrorDB(
        path=tmp_path / "errors.db",
        vector_index_path=tmp_path / "err.index",
        router=router,
    )

    id1 = db.add_error("msg1", type_="t1")
    assert id1 > 0
    assert db.conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        db.add_error("msg1", type_="t1")
    assert db.conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 1
    assert any("Duplicate insert ignored for errors" in r.message for r in caplog.records)

    db.add_error("msg2", type_="t1")
    assert db.conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 2


def test_workflowdb_dedup(tmp_path, caplog, monkeypatch, router):
    from menace import task_handoff_bot as thb
    importlib.reload(thb)

    monkeypatch.setattr(thb.WorkflowDB, "add_embedding", lambda *a, **k: None)
    db = thb.WorkflowDB(
        path=tmp_path / "wf.db",
        vector_index_path=tmp_path / "wf.index",
        router=router,
    )

    wf1 = thb.WorkflowRecord(
        workflow=["a"],
        title="t1",
        description="d1",
        task_sequence=["a"],
    )
    id1 = db.add(wf1)
    assert id1 > 0
    assert db.conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        db.add(
            thb.WorkflowRecord(
                workflow=["a"],
                title="t1",
                description="d1",
                task_sequence=["a"],
            )
        )
    assert db.conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 1
    assert any("Duplicate insert ignored for workflows" in r.message for r in caplog.records)

    db.add(
        thb.WorkflowRecord(
            workflow=["a"],
            title="t2",
            description="d1",
            task_sequence=["a"],
        )
    )
    assert db.conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 2
