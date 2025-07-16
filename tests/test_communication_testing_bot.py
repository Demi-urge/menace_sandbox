import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import sys
from pathlib import Path

import menace.communication_testing_bot as ctb
import menace.mirror_bot as mb

def create_dummy(tmp_path: Path) -> str:
    mod = tmp_path / "dummy_mod.py"
    mod.write_text("""def ping(x=None):\n    return x""")
    sys.path.insert(0, str(tmp_path))
    return "dummy_mod"


def test_functional(tmp_path):
    name = create_dummy(tmp_path)
    ctb.register_module(name)
    db = ctb.CommTestDB(tmp_path / "log.db")
    bot = ctb.CommunicationTestingBot(db=db)
    results = bot.functional_tests([name])
    assert results and results[0].name.startswith(name)
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
    res = bot.integration_test(send, recv, "hi", expected="hi", retries=1, delay=0.01, max_delay=0.02)
    assert res.passed and "expected=hi" in res.details


def test_benchmark_mirror(tmp_path):
    mdb = mb.MirrorDB(tmp_path / "m.db")
    mirror = mb.MirrorBot(mdb)
    mirror.log_interaction("u", "hi", "great")
    mirror.update_style("buddy")
    bot = ctb.CommunicationTestingBot(db=ctb.CommTestDB(":memory:"))
    df = bot.benchmark_mirror(mirror, [("hello", "buddy")])
    assert not df.empty and df.iloc[0]["accuracy"] == 1.0
