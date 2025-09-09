import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "menace.self_test_service",
        ROOT / "self_test_service.py",  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "menace", pkg)
    me = types.SimpleNamespace(Gauge=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "metrics_exporter", me)
    monkeypatch.setitem(sys.modules, "menace.metrics_exporter", me)
    db_mod = types.ModuleType("data_bot")
    db_mod.DataBot = object
    monkeypatch.setitem(sys.modules, "data_bot", db_mod)
    monkeypatch.setitem(sys.modules, "menace.data_bot", db_mod)
    err_db_mod = types.ModuleType("error_bot")
    class _ErrDB:
        def __init__(self, *a, **k):
            pass
    err_db_mod.ErrorDB = _ErrDB
    monkeypatch.setitem(sys.modules, "error_bot", err_db_mod)
    monkeypatch.setitem(sys.modules, "menace.error_bot", err_db_mod)
    err_log_mod = types.ModuleType("error_logger")
    class _ErrLogger:
        def __init__(self, *a, **k):
            pass
    err_log_mod.ErrorLogger = _ErrLogger
    monkeypatch.setitem(sys.modules, "error_logger", err_log_mod)
    monkeypatch.setitem(sys.modules, "menace.error_logger", err_log_mod)
    ae_mod = types.ModuleType("auto_env_setup")
    ae_mod.get_recursive_isolated = lambda: True
    ae_mod.set_recursive_isolated = lambda val: None
    monkeypatch.setitem(sys.modules, "auto_env_setup", ae_mod)
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", ae_mod)
    spec.loader.exec_module(mod)
    return mod


def test_auto_include_isolated_settings(monkeypatch):
    mod = _load_module(monkeypatch)

    class DummySettings:
        auto_include_isolated = True
        recursive_isolated = True

    monkeypatch.setattr(mod, "SandboxSettings", lambda: DummySettings())
    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *a, **k):
            return "", "", {}

    svc = mod.SelfTestService(
        discover_isolated=False,
        recursive_isolated=False,
        context_builder=DummyBuilder(),
    )
    assert svc.discover_isolated is True
    assert svc.recursive_isolated is True
