import logging
import menace.bot_database as bdb
from menace.db_router import init_db_router


def test_add_bot_duplicate(tmp_path, caplog, monkeypatch):
    init_db_router("botdup", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    monkeypatch.setattr(bdb.BotDB, "_embed_record_on_write", lambda *a, **k: None)
    db = bdb.BotDB()

    captured: dict[str, int | None] = {"id": None}
    orig = bdb.insert_if_unique

    def wrapper(*args, **kwargs):
        res = orig(*args, **kwargs)
        captured["id"] = res
        return res

    monkeypatch.setattr(bdb, "insert_if_unique", wrapper)

    rec = bdb.BotRecord(name="b1")
    first = db.add_bot(rec)
    captured["id"] = None
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        second = db.add_bot(bdb.BotRecord(name="b1"))
    assert first == second
    assert captured["id"] is None
    assert "duplicate" in caplog.text.lower()
    assert db.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1
