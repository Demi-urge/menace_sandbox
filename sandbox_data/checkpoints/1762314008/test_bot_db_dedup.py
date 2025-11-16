import menace.db_router as dr
import menace.bot_database as bdb


def test_botdb_duplicate_prevention(tmp_path):
    dr.init_db_router("t", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = bdb.BotDB(tmp_path / "bots.db")
    rec = bdb.BotRecord(name="foo", type_="generic", tasks=["run"], dependencies=["dep"], resources={})
    first = db.add_bot(rec)
    rec2 = bdb.BotRecord(name="foo", type_="generic", tasks=["run"], dependencies=["dep"], resources={})
    second = db.add_bot(rec2)
    assert first == second
    count = db.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0]
    assert count == 1
