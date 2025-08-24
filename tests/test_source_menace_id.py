import pytest
import db_router
import menace.bot_database as bdb


@pytest.fixture
def multi_menace_db(tmp_path, monkeypatch):
    """Create two BotDB instances simulating different menace IDs."""
    # disable embeddings to avoid heavy dependencies
    monkeypatch.setattr(bdb.BotDB, "_embed_record_on_write", lambda *a, **k: None)
    shared = tmp_path / "shared.db"

    # first menace "alpha"
    router_a = db_router.init_db_router("alpha", str(tmp_path / "alpha.db"), str(shared))
    bdb.router = router_a
    db_a = bdb.BotDB(tmp_path / "a.db")
    id_a = db_a.add_bot(bdb.BotRecord(name="a"))

    # second menace "beta"
    router_b = db_router.init_db_router("beta", str(tmp_path / "beta.db"), str(shared))
    bdb.router = router_b
    db_b = bdb.BotDB(tmp_path / "b.db")
    id_b = db_b.add_bot(bdb.BotRecord(name="b"))

    return {
        "router_a": router_a,
        "router_b": router_b,
        "db_a": db_a,
        "db_b": db_b,
        "id_a": id_a,
        "id_b": id_b,
    }


def test_inserts_populate_source_menace_id(multi_menace_db):
    db_a = multi_menace_db["db_a"]
    db_b = multi_menace_db["db_b"]
    id_a = multi_menace_db["id_a"]
    id_b = multi_menace_db["id_b"]

    src_a = db_a.conn.execute("SELECT source_menace_id FROM bots WHERE id=?", (id_a,)).fetchone()[0]
    src_b = db_b.conn.execute("SELECT source_menace_id FROM bots WHERE id=?", (id_b,)).fetchone()[0]
    assert src_a == "alpha"
    assert src_b == "beta"


def test_default_queries_return_only_current_menace(multi_menace_db):
    bdb.router = multi_menace_db["router_a"]
    db_a = multi_menace_db["db_a"]
    names = {r["name"] for r in db_a.fetch_all()}
    assert names == {"a"}


def test_explicit_local_scope(multi_menace_db):
    bdb.router = multi_menace_db["router_a"]
    db_a = multi_menace_db["db_a"]
    names = {r["name"] for r in db_a.fetch_all(scope="local")}
    assert names == {"a"}


def test_scope_selection(multi_menace_db):
    bdb.router = multi_menace_db["router_a"]
    db_a = multi_menace_db["db_a"]

    all_names = {r["name"] for r in db_a.fetch_all(scope="all")}
    assert all_names == {"a", "b"}

    global_names = {r["name"] for r in db_a.fetch_all(scope="global")}
    assert global_names == {"b"}

    beta_names = [r["name"] for r in db_a.fetch_all(source_menace_id="beta")]
    assert beta_names == ["b"]
