import sys
from pathlib import Path

import menace.bot_testing_bot as btb
import db_router


def create_dummy(tmp_path: Path) -> str:
    mod = tmp_path / "dummy.py"  # path-ignore
    mod.write_text(
        """from __future__ import annotations

def hello(name: str = 'x'):
    return f'hi {name}'
"""
    )
    sys.path.insert(0, str(tmp_path))
    return "dummy"


def test_run_unit(tmp_path):
    name = create_dummy(tmp_path)
    router = db_router.init_db_router(
        "bot_testing",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )
    db = btb.TestingLogDB(connection_factory=lambda: router.get_connection("results"))
    bot = btb.BotTestingBot(db)
    results = bot.run_unit_tests([name], parallel=False)
    assert results
    stored = db.all()
    assert stored
    assert all(r.bot == name for r in stored)
    assert all(r.code_hash for r in stored)


def create_imports(tmp_path: Path) -> str:
    mod_a = tmp_path / "defs.py"  # path-ignore
    mod_a.write_text(
        """calls = []

def ext_func():
    calls.append('ext')

class ExtCls:
    def method(self):
        calls.append('ext_method')
"""
    )

    mod_b = tmp_path / "use_defs.py"  # path-ignore
    mod_b.write_text(
        """from defs import ext_func, ExtCls

calls = []

def local_func():
    calls.append('local')

class LocalCls:
    def method(self):
        calls.append('local_method')
"""
    )

    sys.path.insert(0, str(tmp_path))
    return "use_defs"


def test_run_unit_skips_imports(tmp_path):
    name = create_imports(tmp_path)
    router = db_router.init_db_router(
        "bot_testing2",
        local_db_path=str(tmp_path / "local2.db"),
        shared_db_path=str(tmp_path / "shared2.db"),
    )
    db = btb.TestingLogDB(connection_factory=lambda: router.get_connection("results"))
    bot = btb.BotTestingBot(db)
    results = bot.run_unit_tests([name], parallel=False)
    assert results

    defs = __import__("defs")
    use_defs = __import__(name)

    # imported functions/classes should not be executed
    assert defs.calls == []
    assert use_defs.calls
