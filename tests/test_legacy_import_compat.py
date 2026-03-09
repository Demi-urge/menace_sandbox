from __future__ import annotations

import sys
import types


def test_legacy_import_symbols_resolve(monkeypatch) -> None:
    # Some legacy import paths transitively touch optional SQLAlchemy helpers;
    # stub them so compatibility imports remain import-time safe in minimal envs.
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    sqlalchemy_engine_mod = types.ModuleType("sqlalchemy.engine")
    sqlalchemy_engine_mod.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_mod)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqlalchemy_engine_mod)

    from menace.diagnostic_manager import ContextBuilder
    from coding_bot_interface import prepare_pipeline_for_bootstrap
    from menace.self_coding_manager import DataBot
    from menace.self_debugger_sandbox import SelfDebuggerSandbox

    assert callable(prepare_pipeline_for_bootstrap)
    assert ContextBuilder is not None
    assert DataBot is not None
    assert SelfDebuggerSandbox is not None
