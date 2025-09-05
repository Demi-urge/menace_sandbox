import sys
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
import menace.communication_testing_bot as ctb
import menace.db_router as db_router


@settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True))
def test_functional_random_module(tmp_path, name):
    mod = tmp_path / f"{name}.py"  # path-ignore
    mod.write_text("def ping(x=None):\n    return x")
    sys.path.insert(0, str(tmp_path))
    ctb.register_module(name)
    router = db_router.DBRouter(
        "ct", str(tmp_path / "log.db"), str(tmp_path / "log.db")
    )
    db = ctb.CommTestDB(tmp_path / "log.db", router=router)
    bot = ctb.CommunicationTestingBot(db=db)
    results = bot.functional_tests([name])
    assert results and results[0].passed
    assert db.fetch_failed() == []
    ctb.unregister_module(name)


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(min_size=1, max_size=20))
def test_integration_random_message(tmp_path, message):
    store = []
    def send(m):
        store.append(m)
    def recv():
        return store.pop(0)
    router = db_router.DBRouter(
        "ct", str(tmp_path / "log.db"), str(tmp_path / "log.db")
    )
    bot = ctb.CommunicationTestingBot(
        db=ctb.CommTestDB(tmp_path / "log.db", router=router)
    )
    res = bot.integration_test(send, recv, message, expected=message)
    assert res.passed
