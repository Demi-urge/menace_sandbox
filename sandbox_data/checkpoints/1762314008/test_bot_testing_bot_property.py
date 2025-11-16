import sys

from hypothesis import given, strategies as st, settings, HealthCheck

import menace.bot_testing_bot as btb
import db_router


@settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True))
def test_run_unit_random_module(tmp_path, name):
    mod = tmp_path / f"{name}.py"  # path-ignore
    mod.write_text("""def hello(name='x'):\n    return f'hi {name}'\n""")
    sys.path.insert(0, str(tmp_path))
    router = db_router.init_db_router(
        "bot_testing_prop",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )
    db = btb.TestingLogDB(connection_factory=lambda: router.get_connection("results"))
    bot = btb.BotTestingBot(db)
    results = bot.run_unit_tests([name], parallel=False)
    assert results and all(r.passed for r in results)
    assert db.all()
