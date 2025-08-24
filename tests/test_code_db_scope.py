import db_router
import menace.code_database as cdb


def _setup_code_db(tmp_path):
    shared = tmp_path / "shared.db"
    # disable embedding to keep tests lightweight
    cdb.CodeDB._embed_record_on_write = lambda *a, **k: None
    router_a = db_router.init_db_router("alpha", str(tmp_path / "alpha.db"), str(shared))
    db_a = cdb.CodeDB(tmp_path / "a.db", router=router_a)
    id_a = db_a.add(cdb.CodeRecord(code="print('a')", summary="A", complexity_score=1.0))
    db_a.link_bot(id_a, "shared")

    router_b = db_router.init_db_router("beta", str(tmp_path / "beta.db"), str(shared))
    db_b = cdb.CodeDB(tmp_path / "b.db", router=router_b)
    id_b = db_b.add(cdb.CodeRecord(code="print('b')", summary="B", complexity_score=2.0))
    db_b.link_bot(id_b, "shared")

    return {
        "router_a": router_a,
        "router_b": router_b,
        "db_a": db_a,
        "db_b": db_b,
    }


def test_fetch_all_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    assert {r["summary"] for r in db_a.fetch_all(scope="local")} == {"A"}
    assert {r["summary"] for r in db_a.fetch_all(scope="global")} == {"B"}
    assert {r["summary"] for r in db_a.fetch_all(scope="all")} == {"A", "B"}


def test_by_complexity_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    assert [r["summary"] for r in db_a.by_complexity(scope="local")] == ["A"]
    assert [r["summary"] for r in db_a.by_complexity(scope="global")] == ["B"]
    assert {r["summary"] for r in db_a.by_complexity(scope="all")} == {"A", "B"}


def test_search_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    assert [r["summary"] for r in db_a.search("print", scope="local")] == ["A"]
    assert [r["summary"] for r in db_a.search("print", scope="global")] == ["B"]
    assert {r["summary"] for r in db_a.search("print", scope="all")} == {"A", "B"}


def test_codes_for_bot_scopes(tmp_path):
    ctx = _setup_code_db(tmp_path)
    db_a = ctx["db_a"]

    ids_local = set(db_a.codes_for_bot("shared", scope="local"))
    ids_global = set(db_a.codes_for_bot("shared", scope="global"))
    ids_all = set(db_a.codes_for_bot("shared", scope="all"))
    assert ids_all == ids_local | ids_global
