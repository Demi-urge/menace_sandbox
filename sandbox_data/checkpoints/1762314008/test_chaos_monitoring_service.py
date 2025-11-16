import importlib.util
import sys
import types
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp())

pkg = types.ModuleType("menace")
pkg.__path__ = [str(TMP), str(ROOT)]
pkg.RAISE_ERRORS = False
sys.modules["menace"] = pkg

# stub vector service and ContextBuilder
vector_service_pkg = types.ModuleType("vector_service")
context_mod = types.ModuleType("vector_service.context_builder")


class ContextBuilder:
    def __init__(self) -> None:
        self.refreshed = False

    def refresh_db_weights(self) -> None:
        self.refreshed = True


context_mod.ContextBuilder = ContextBuilder
vector_service_pkg.context_builder = context_mod
sys.modules["vector_service"] = vector_service_pkg
sys.modules["vector_service.context_builder"] = context_mod

# stub modules
(TMP / "chaos_scheduler.py").write_text(  # path-ignore
    (
        "class ChaosScheduler:\n"
        "    def __init__(self, watchdog=None):\n"
        "        self.watchdog=watchdog\n"
        "        self.interval=0\n"
        "        self.started=False\n"
        "    def start(self):\n"
        "        self.started=True\n"
    )
)
(TMP / "watchdog.py").write_text(  # path-ignore
    "class Watchdog:\n    def __init__(self, *a, **k):\n        self.synthetic_faults=[]\n"
)
(TMP / "error_bot.py").write_text("class ErrorDB: pass\n")  # path-ignore
(TMP / "resource_allocation_optimizer.py").write_text("class ROIDB: pass\n")  # path-ignore
(TMP / "data_bot.py").write_text("class MetricsDB: pass\n")  # path-ignore
(TMP / "advanced_error_management.py").write_text(  # path-ignore
    (
        "class AutomatedRollbackManager:\n"
        "    def __init__(self, raise_error=False):\n"
        "        self.raise_error=raise_error\n"
        "        self.calls=[]\n"
        "    def auto_rollback(self, version, bots):\n"
        "        self.calls.append((version, bots))\n"
        "        if self.raise_error:\n"
        "            raise RuntimeError('boom')\n"
    )
)
class DummyBuilder(ContextBuilder):
    pass


class BrokenBuilder(ContextBuilder):
    def refresh_db_weights(self) -> None:  # pragma: no cover - trigger error
        raise RuntimeError("boom")


spec = importlib.util.spec_from_file_location(
    "menace.chaos_monitoring_service",
    ROOT / "chaos_monitoring_service.py",  # path-ignore
    submodule_search_locations=[str(TMP), str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["menace.chaos_monitoring_service"] = mod
spec.loader.exec_module(mod)


def _run_once(svc):
    class OnceEvent:
        def __init__(self):
            self.count = 0

        def wait(self, interval):
            self.count += 1
            return self.count > 1

    svc._monitor(OnceEvent())


def test_auto_rollback_triggered():
    wd = mod.Watchdog()
    wd.synthetic_faults = [{"bot": "x"}]
    sched = mod.ChaosScheduler(wd)
    rb = mod.AutomatedRollbackManager()
    svc = mod.ChaosMonitoringService(
        scheduler=sched, rollback_mgr=rb, context_builder=DummyBuilder()
    )
    _run_once(svc)
    assert rb.calls == [("latest", ["x"])]


def test_no_action_when_recovered():
    wd = mod.Watchdog()
    wd.synthetic_faults = [{"bot": "x", "recovered": True}]
    sched = mod.ChaosScheduler(wd)
    rb = mod.AutomatedRollbackManager()
    svc = mod.ChaosMonitoringService(
        scheduler=sched, rollback_mgr=rb, context_builder=DummyBuilder()
    )
    _run_once(svc)
    assert rb.calls == []


def test_error_logged(caplog):
    wd = mod.Watchdog()
    wd.synthetic_faults = [{"bot": "x"}]
    sched = mod.ChaosScheduler(wd)
    rb = mod.AutomatedRollbackManager(raise_error=True)
    svc = mod.ChaosMonitoringService(
        scheduler=sched, rollback_mgr=rb, context_builder=DummyBuilder()
    )
    caplog.set_level("ERROR")
    _run_once(svc)
    assert "auto rollback failed" in caplog.text


def test_weights_refreshed_on_startup():
    builder = DummyBuilder()
    mod.ChaosMonitoringService(context_builder=builder)
    assert builder.refreshed


def test_requires_context_builder_instance():
    with pytest.raises(TypeError):
        mod.ChaosMonitoringService(context_builder=object())


def test_refresh_errors_surface_early():
    with pytest.raises(RuntimeError):
        mod.ChaosMonitoringService(context_builder=BrokenBuilder())
