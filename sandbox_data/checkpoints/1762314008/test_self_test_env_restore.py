import importlib.util
import sys
import os
import types
import asyncio
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_service(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    pkg.RAISE_ERRORS = False
    monkeypatch.setitem(sys.modules, "menace", pkg)

    class DummyGauge:
        def inc(self, *a, **k):
            pass

        def set(self, *a, **k):
            pass

    me = types.SimpleNamespace(
        Gauge=lambda *a, **k: DummyGauge(),
        orphan_modules_reintroduced_total=DummyGauge(),
        orphan_modules_passed_total=DummyGauge(),
        orphan_modules_failed_total=DummyGauge(),
        orphan_modules_tested_total=DummyGauge(),
        orphan_modules_reclassified_total=DummyGauge(),
        orphan_modules_redundant_total=DummyGauge(),
        orphan_modules_legacy_total=DummyGauge(),
    )
    monkeypatch.setitem(sys.modules, "metrics_exporter", me)
    monkeypatch.setitem(sys.modules, "menace.metrics_exporter", me)

    data_bot_mod = types.ModuleType("data_bot")
    data_bot_mod.DataBot = object
    monkeypatch.setitem(sys.modules, "data_bot", data_bot_mod)
    monkeypatch.setitem(sys.modules, "menace.data_bot", data_bot_mod)

    error_bot_mod = types.ModuleType("error_bot")
    error_bot_mod.ErrorDB = object
    monkeypatch.setitem(sys.modules, "error_bot", error_bot_mod)
    monkeypatch.setitem(sys.modules, "menace.error_bot", error_bot_mod)

    error_logger_mod = types.ModuleType("error_logger")

    class DummyLogger:
        def __init__(self, db=None):
            pass

        def log(self, *a, **k):
            pass

    error_logger_mod.ErrorLogger = DummyLogger
    monkeypatch.setitem(sys.modules, "error_logger", error_logger_mod)
    monkeypatch.setitem(sys.modules, "menace.error_logger", error_logger_mod)

    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, "menace.self_test_service", mod)

    monkeypatch.setattr(mod, "load_orphan_cache", lambda *a, **k: {})
    monkeypatch.setattr(mod, "append_orphan_cache", lambda *a, **k: None)
    monkeypatch.setattr(mod, "append_orphan_classifications", lambda *a, **k: None)
    monkeypatch.setattr(mod, "collect_local_dependencies", lambda *a, **k: set())
    monkeypatch.setattr(mod.SelfTestService, "_discover_orphans", lambda self: [])

    async def fake_test(self, paths):
        return set(), set()

    monkeypatch.setattr(mod.SelfTestService, "_test_orphan_modules", fake_test)
    monkeypatch.setattr(mod.SelfTestService, "_save_state", lambda self, *a, **k: None)
    return mod.SelfTestService


def test_env_vars_restored_on_error(monkeypatch):
    Service = _load_service(monkeypatch)
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "orig")
    monkeypatch.delenv("SELF_TEST_DISCOVER_ORPHANS", raising=False)

    def boom(repo):
        raise RuntimeError("boom")

    mod = sys.modules["menace.self_test_service"]
    monkeypatch.setattr(mod, "load_orphan_cache", boom)

    svc = Service(pytest_args="")
    with pytest.raises(RuntimeError):
        asyncio.run(svc._run_once())
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "orig"
    assert os.getenv("SELF_TEST_DISCOVER_ORPHANS") is None
