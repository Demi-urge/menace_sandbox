import logging
import menace.bot_database as bdb
from menace.db_router import init_db_router


def test_add_bot_duplicate(tmp_path, caplog, monkeypatch):
    init_db_router("botdup", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    monkeypatch.setattr(bdb.BotDB, "_embed_record_on_write", lambda *a, **k: None)
    db = bdb.BotDB()
    rec = bdb.BotRecord(name="b1")
    with caplog.at_level(logging.WARNING):
        first = db.add_bot(rec)
        second = db.add_bot(bdb.BotRecord(name="b1"))
    assert first == second
    assert "duplicate" in caplog.text.lower()
    assert db.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1
