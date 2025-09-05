import importlib.util
import sys
import os
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _load_module(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    pkg.RAISE_ERRORS = False
    monkeypatch.setitem(sys.modules, "menace", pkg)
    # stub metrics exporter to avoid global registry conflicts
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
    # stub modules with heavy dependencies
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
    monkeypatch.setitem(sys.modules, "menace.self_test_service", mod)
    spec.loader.exec_module(mod)

    class DummyService:
        def __init__(self, *a, **k):
            self.results = []

        async def _run_once(self, refresh_orphans=False):
            return 0

    monkeypatch.setattr(mod, "SelfTestService", DummyService)
    return mod


def test_cli_auto_include_sets_env(monkeypatch):
    mod = _load_module(monkeypatch)
    for var in [
        "SANDBOX_AUTO_INCLUDE_ISOLATED",
        "SANDBOX_RECURSIVE_ISOLATED",
        "SANDBOX_DISCOVER_ISOLATED",
        "SELF_TEST_AUTO_INCLUDE_ISOLATED",
        "SELF_TEST_RECURSIVE_ISOLATED",
        "SELF_TEST_DISCOVER_ISOLATED",
    ]:
        monkeypatch.delenv(var, raising=False)
    mod.cli(["run", "--auto-include-isolated"])
    assert os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED") == "1"
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_AUTO_INCLUDE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_DISCOVER_ISOLATED") == "1"


def test_cli_recursive_include_toggles_env(monkeypatch):
    mod = _load_module(monkeypatch)
    for var in ["SANDBOX_RECURSIVE_ORPHANS", "SELF_TEST_RECURSIVE_ORPHANS"]:
        monkeypatch.delenv(var, raising=False)
    mod.cli(["run", "--no-recursive-include"])
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "0"
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "0"
    mod.cli(["run", "--recursive-include"])
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "1"
