import pytest
import sys
from pathlib import Path
import menace.communication_testing_bot as ctb
import menace.mirror_bot as mb
import menace.db_router as db_router


def create_dummy(tmp_path: Path) -> str:
    mod = tmp_path / "dummy_mod.py"  # path-ignore
    mod.write_text("""def ping(x=None):\n    return x""")
    sys.path.insert(0, str(tmp_path))
    return "dummy_mod"


def test_functional(tmp_path):
    name = create_dummy(tmp_path)
    ctb.register_module(name)
    router = db_router.DBRouter(
        "ct", str(tmp_path / "log.db"), str(tmp_path / "log.db")
    )
    db = ctb.CommTestDB(tmp_path / "log.db", router=router)
    bot = ctb.CommunicationTestingBot(db=db)
    results = bot.functional_tests([name])
    assert results and results[0].name
    rep = bot.report(output="json")
    assert "passed" in rep
    same = db.fetch_by_name(results[0].name)
    assert same and same[0].name == results[0].name
    assert db.fetch_failed() == []


def test_integration():
    store: list[str] = []

    def send(msg: str) -> None:
        store.append(msg)

    def recv() -> str:
        return store.pop(0)

    bot = ctb.CommunicationTestingBot(db=ctb.CommTestDB(":memory:"))
    res = bot.integration_test(
        send,
        recv,
        "hi",
        expected="hi",
        retries=1,
        delay=0.01,
        max_delay=0.02,
    )
    assert res.passed and "expected=hi" in res.details


def test_benchmark_mirror(tmp_path):
    pytest.importorskip("pandas")
    router = db_router.init_db_router(
        "mirror", str(tmp_path / "m.db"), str(tmp_path / "m.db")
    )
    mdb = mb.MirrorDB(router=router)
    mirror = mb.MirrorBot(mdb)
    mirror.log_interaction("u", "hi", "great")
    mirror.update_style("buddy")
    db_router_obj = db_router.DBRouter(
        "ct", str(tmp_path / "log.db"), str(tmp_path / "log.db")
    )
    bot = ctb.CommunicationTestingBot(
        db=ctb.CommTestDB(tmp_path / "log.db", router=db_router_obj)
    )
    df = bot.benchmark_mirror(mirror, [("hello", "buddy")])
    assert not df.empty and df.iloc[0]["accuracy"] == 1.0


def test_fetch_scope(tmp_path):
    router = db_router.DBRouter(
        "ct", str(tmp_path / "log.db"), str(tmp_path / "log.db")
    )
    db = ctb.CommTestDB(tmp_path / "log.db", router=router)
    local = ctb.CommTestResult(name="local", passed=True, details="ok")
    db.log(local)
    foreign = ctb.CommTestResult(name="foreign", passed=False, details="no")
    db.log(foreign, source_menace_id="other")

    loc = db.fetch(scope="local")
    assert [r.name for r in loc] == ["local"]

    glob = db.fetch(scope="global")
    assert [r.name for r in glob] == ["foreign"]

    all_rows = db.fetch(scope="all")
    assert [r.name for r in all_rows] == ["local", "foreign"]
