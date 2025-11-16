# flake8: noqa
import types
import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.menace_orchestrator as mo

class DummyWatchdog:
    def __init__(self):
        self.hb = []
        self.healed = []
        self.healer = types.SimpleNamespace(heal=lambda b, pid=None: self.healed.append(b))
    def record_heartbeat(self, name):
        self.hb.append(name)
    def schedule(self, interval=60):
        pass
    def _check(self, timeout=0):
        for bot in list(self.hb):
            self.healer.heal(bot)

@pytest.fixture
def orch(monkeypatch):
    monkeypatch.setenv("TRENDING_SCAN_INTERVAL", "0")
    monkeypatch.setenv("LEARNING_INTERVAL", "0")
    monkeypatch.setenv("WATCHDOG_INTERVAL", "0")
    monkeypatch.setattr(mo, "BackgroundScheduler", None)
    monkeypatch.setattr(mo.TrendingScraper, "scrape_reddit", lambda self: None)
    monkeypatch.setattr(mo, "learning_main", lambda *a, **k: None)
    o = mo.MenaceOrchestrator(context_builder=mo.ContextBuilder())
    o.watchdog = DummyWatchdog()
    return o

def test_start_and_restart(monkeypatch, orch):
    monkeypatch.setattr(mo._SimpleScheduler, "add_job", lambda self, func, interval, id: self.tasks.setdefault(id, (interval, func, 0.0)))
    monkeypatch.setattr(mo._SimpleScheduler, "_run", lambda self: [info[1]() for info in self.tasks.values()] or (_ for _ in ()).throw(SystemExit))
    orch.start_scheduled_jobs()
    with pytest.raises(SystemExit):
        orch.scheduler._run()
    assert set(orch.watchdog.hb) == {"trending_scan", "learning", "planning"}
    orch.watchdog.hb = []
    orch.watchdog._check()
    assert set(orch.watchdog.healed) == {"trending_scan", "learning", "planning"}
