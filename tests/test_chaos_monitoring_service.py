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

# stub modules
(TMP / "chaos_scheduler.py").write_text(  # path-ignore
    """class ChaosScheduler:\n    def __init__(self, watchdog=None):\n        self.watchdog=watchdog\n        self.interval=0\n        self.started=False\n    def start(self):\n        self.started=True\n"""
)
(TMP / "watchdog.py").write_text(  # path-ignore
    """class Watchdog:\n    def __init__(self, *a, **k):\n        self.synthetic_faults=[]\n"""
)
(TMP / "error_bot.py").write_text("class ErrorDB: pass\n")  # path-ignore
(TMP / "resource_allocation_optimizer.py").write_text("class ROIDB: pass\n")  # path-ignore
(TMP / "data_bot.py").write_text("class MetricsDB: pass\n")  # path-ignore
(TMP / "advanced_error_management.py").write_text(  # path-ignore
    """class AutomatedRollbackManager:\n    def __init__(self, raise_error=False):\n        self.raise_error=raise_error\n        self.calls=[]\n    def auto_rollback(self, version, bots):\n        self.calls.append((version, bots))\n        if self.raise_error:\n            raise RuntimeError('boom')\n"""
)

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
            self.count=0
        def wait(self, interval):
            self.count+=1
            return self.count>1
    svc._monitor(OnceEvent())


def test_auto_rollback_triggered():
    wd = mod.Watchdog()
    wd.synthetic_faults = [{"bot": "x"}]
    sched = mod.ChaosScheduler(wd)
    rb = mod.AutomatedRollbackManager()
    svc = mod.ChaosMonitoringService(scheduler=sched, rollback_mgr=rb)
    _run_once(svc)
    assert rb.calls == [("latest", ["x"])]


def test_no_action_when_recovered():
    wd = mod.Watchdog()
    wd.synthetic_faults = [{"bot": "x", "recovered": True}]
    sched = mod.ChaosScheduler(wd)
    rb = mod.AutomatedRollbackManager()
    svc = mod.ChaosMonitoringService(scheduler=sched, rollback_mgr=rb)
    _run_once(svc)
    assert rb.calls == []


def test_error_logged(caplog):
    wd = mod.Watchdog()
    wd.synthetic_faults = [{"bot": "x"}]
    sched = mod.ChaosScheduler(wd)
    rb = mod.AutomatedRollbackManager(raise_error=True)
    svc = mod.ChaosMonitoringService(scheduler=sched, rollback_mgr=rb)
    caplog.set_level("ERROR")
    _run_once(svc)
    assert "auto rollback failed" in caplog.text


def test_builder_required_when_scheduler_missing():
    with pytest.raises(ValueError):
        mod.ChaosMonitoringService()
