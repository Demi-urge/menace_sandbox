import importlib
import logging

import pytest

import db_router
from db_dedup import hash_fields, insert_if_unique


@pytest.fixture
def router(tmp_path):
    """Initialize a fresh DBRouter pointing at temporary files."""
    path = tmp_path / "db.sqlite"
    return db_router.init_db_router("test", str(path), str(path))


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
        id2 = db.add_bot(bdb.BotRecord(name="alpha", purpose="p"))
    # Duplicate should reuse the original id and not create a new row
    assert id2 == id1
    assert db.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1
    assert any("Duplicate insert ignored for bots" in r.message for r in caplog.records)

    id3 = db.add_bot(bdb.BotRecord(name="beta", purpose="p"))
    assert id3 != id1
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

    # ``content_hash`` column may not be added on older SQLite versions; ensure
    # it exists so deduplication can be tested reliably.
    try:
        db.conn.execute(
            "ALTER TABLE enhancements ADD COLUMN content_hash TEXT NOT NULL"
        )
    except Exception:
        pass
    db.conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "idx_enhancements_content_hash ON enhancements(content_hash)"
    )
    db.conn.commit()

    enh1 = ceb.Enhancement(idea="idea1", rationale="r1")
    id1 = db.add(enh1)
    assert id1 > 0
    assert db.conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        id2 = db.add(ceb.Enhancement(idea="idea1", rationale="r1"))
    assert id2 == id1
    assert db.conn.execute("SELECT COUNT(*) FROM enhancements").fetchone()[0] == 1
    assert any("duplicate" in r.message.lower() for r in caplog.records)

    id3 = db.add(ceb.Enhancement(idea="idea2", rationale="r1"))
    assert id3 != id1
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
        id2 = db.add_error("msg1", type_="t1")
    assert id2 == id1
    assert db.conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 1
    assert any("Duplicate insert ignored for errors" in r.message for r in caplog.records)

    id3 = db.add_error("msg2", type_="t1")
    assert id3 != id1
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
        id2 = db.add(
            thb.WorkflowRecord(
                workflow=["a"],
                title="t1",
                description="d1",
                task_sequence=["a"],
            )
        )
    assert id2 == id1
    assert db.conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 1
    assert any("Duplicate insert ignored for workflows" in r.message for r in caplog.records)

    id3 = db.add(
        thb.WorkflowRecord(
            workflow=["a"],
            title="t2",
            description="d2",
            task_sequence=["a"],
        )
    )
    assert id3 != id1
    assert db.conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0] == 2


def test_hash_fields_deterministic():
    data1 = {"a": 1, "b": 2}
    data2 = {"b": 2, "a": 1}
    assert hash_fields(data1, ["a", "b"]) == hash_fields(data2, ["a", "b"])


def test_hash_fields_missing_key():
    data = {"a": 1}
    with pytest.raises(KeyError, match="Missing fields for hashing: b"):
        hash_fields(data, ["a", "b"])


def test_insert_if_unique_duplicate_returns_existing_id(tmp_path, caplog):
    path = tmp_path / "dedup.sqlite"
    router = db_router.init_db_router("test", str(path), str(path))
    conn = router.local_conn
    conn.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE NOT NULL)"
    )
    logger = logging.getLogger(__name__)

    id1 = insert_if_unique(
        "items",
        {"name": "alpha"},
        ["name"],
        "m1",
        conn=conn,
        logger=logger,
    )
    assert id1 == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        id2 = insert_if_unique(
            "items",
            {"name": "alpha"},
            ["name"],
            "m1",
            conn=conn,
            logger=logger,
        )
    assert id2 == id1
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1
    assert any(
        "Duplicate insert ignored for items" in r.message for r in caplog.records
    )


def test_insert_if_unique_missing_field_sqlite(tmp_path):
    path = tmp_path / "dedup.sqlite"
    router = db_router.init_db_router("test", str(path), str(path))
    conn = router.local_conn
    conn.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE NOT NULL)"
    )
    logger = logging.getLogger(__name__)

    with pytest.raises(KeyError, match="Missing fields for hashing: name"):
        insert_if_unique(
            "items",
            {},
            ["name"],
            "m1",
            conn=conn,
            logger=logger,
        )
