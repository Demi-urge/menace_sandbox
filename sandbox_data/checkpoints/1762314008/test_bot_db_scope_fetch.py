import menace.bot_database as bdb
from menace.db_router import init_db_router

def test_botdb_fetch_all_scopes(tmp_path, monkeypatch):
    shared = tmp_path / "shared.db"
    monkeypatch.setattr(bdb.BotDB, "_embed_record_on_write", lambda *a, **k: None)

    router_a = init_db_router("one", str(tmp_path / "one.db"), str(shared))
    bdb.router = router_a
    db_a = bdb.BotDB(tmp_path / "a.db")
    db_a.add_bot(bdb.BotRecord(name="a"))

    router_b = init_db_router("two", str(tmp_path / "two.db"), str(shared))
    bdb.router = router_b
    db_b = bdb.BotDB(tmp_path / "b.db")
    db_b.add_bot(bdb.BotRecord(name="b"))

    bdb.router = router_a
    assert {r["name"] for r in db_a.fetch_all(scope="local")} == {"a"}
    assert {r["name"] for r in db_a.fetch_all(scope="global")} == {"b"}
    assert {r["name"] for r in db_a.fetch_all(scope="all")} == {"a", "b"}
