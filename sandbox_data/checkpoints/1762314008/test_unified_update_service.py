import os
import subprocess
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace.unified_update_service import UnifiedUpdateService
from menace.unified_event_bus import UnifiedEventBus

class DummyUpdate:
    def __init__(self):
        self.calls = []
    def _run_once(self, verify_host=None):
        self.calls.append(verify_host)

class DummyDeployer:
    def __init__(self, bus):
        self.event_bus = bus
        self.updated = []
        self.deploy_called = False
    def deploy(self, name, bots, spec):
        self.deploy_called = True
    def auto_update_nodes(self, nodes, branch="main"):
        self.updated.append(list(nodes))
        for n in nodes:
            self.event_bus.publish("nodes:update", {"node": n, "status": "success"})

class FailingDeployer(DummyDeployer):
    def auto_update_nodes(self, nodes, branch="main"):
        self.updated.append(list(nodes))
        for n in nodes:
            self.event_bus.publish("nodes:update", {"node": n, "status": "failed"})

class DummyRollback:
    def __init__(self):
        self.calls = []
    def auto_rollback(self, tag, nodes):
        self.calls.append((tag, nodes))


def _fake_run(cmd, check=False):
    return subprocess.CompletedProcess(cmd, 0)


def test_staged_rollout(monkeypatch):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = DummyDeployer(bus)
    svc = UnifiedUpdateService(updater_service=upd, deployer=dep)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("NODES", "a,b,c")
    monkeypatch.setenv("ROLLOUT_BATCH_SIZE", "2")
    monkeypatch.setenv("DEP_VERIFY_HOST", "vh")
    svc._cycle()
    assert upd.calls == ["vh"]
    assert dep.updated == [["a", "b"], ["c"]]


def test_rollout_failure_triggers_rollback(monkeypatch):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = FailingDeployer(bus)
    rb = DummyRollback()
    svc = UnifiedUpdateService(updater_service=upd, deployer=dep, rollback_mgr=rb)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("NODES", "a,b")
    monkeypatch.setenv("ROLLOUT_BATCH_SIZE", "1")
    monkeypatch.setenv("DEP_VERIFY_HOST", "vh")
    svc._cycle()
    assert dep.updated == [["a"]]
    assert rb.calls == [("latest", ["a", "b"])]


class FailingRollback(DummyRollback):
    def auto_rollback(self, tag, nodes):
        super().auto_rollback(tag, nodes)
        raise RuntimeError("rollback boom")


def test_rollback_errors_logged(monkeypatch, caplog):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = FailingDeployer(bus)
    rb = FailingRollback()
    svc = UnifiedUpdateService(
        updater_service=upd,
        deployer=dep,
        rollback_mgr=rb,
        max_retries=1,
    )
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("NODES", "a")
    monkeypatch.setenv("ROLLOUT_BATCH_SIZE", "1")
    monkeypatch.setenv("DEP_VERIFY_HOST", "vh")
    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError):
        svc._cycle()
    assert "rollback failed" in caplog.text
