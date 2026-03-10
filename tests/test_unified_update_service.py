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
        self.deploy_spec = None
    def deploy(self, name, bots, spec):
        self.deploy_called = True
        self.deploy_spec = spec
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


def _pytest_collection_failure(cmd, check=False):
    if cmd[:2] == ["pytest", "-q"]:
        raise subprocess.CalledProcessError(2, cmd)
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


def test_cycle_constructs_deployment_spec_with_required_fields(monkeypatch):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = DummyDeployer(bus)
    svc = UnifiedUpdateService(updater_service=upd, deployer=dep)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.delenv("NODES", raising=False)
    svc._cycle()
    assert dep.deploy_called is True
    assert dep.deploy_spec.name == "auto-update"
    assert dep.deploy_spec.env == {}
    assert dep.deploy_spec.resources == {}


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


def test_run_continuous_recovers_when_spec_construction_fails(monkeypatch, caplog):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = DummyDeployer(bus)
    svc = UnifiedUpdateService(updater_service=upd, deployer=dep)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    from menace import unified_update_service as uus

    class BrokenSpec:
        def __init__(self, *args, **kwargs):
            raise ValueError("bad spec")

    monkeypatch.setattr(uus, "DeploymentSpec", BrokenSpec)

    import threading

    stop = threading.Event()
    caplog.set_level("ERROR")

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target
        def start(self):
            self.target()

    monkeypatch.setattr(threading, "Thread", ImmediateThread)
    monkeypatch.setattr(stop, "wait", lambda _interval: True)

    svc.run_continuous(interval=0.01, stop_event=stop)
    assert "failed to build deployment spec" in caplog.text
    assert "cycle failed and will be retried on next interval" in caplog.text


def test_cycle_uses_smoke_scope_by_default(monkeypatch):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = DummyDeployer(bus)
    svc = UnifiedUpdateService(updater_service=upd, deployer=dep)
    commands = []

    def _capture_run(cmd, check=False):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _capture_run)
    monkeypatch.delenv("UNIFIED_UPDATE_FULL_SUITE", raising=False)
    svc._cycle()
    assert ["pytest", "-q", "-m", "smoke"] in commands


def test_cycle_uses_full_suite_when_flag_set(monkeypatch):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = DummyDeployer(bus)
    svc = UnifiedUpdateService(updater_service=upd, deployer=dep)
    commands = []

    def _capture_run(cmd, check=False):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _capture_run)
    monkeypatch.setenv("UNIFIED_UPDATE_FULL_SUITE", "true")
    svc._cycle()
    assert ["pytest", "-q"] in commands


def test_collection_failure_is_terminal_and_skips_retry(monkeypatch, caplog):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = DummyDeployer(bus)
    svc = UnifiedUpdateService(updater_service=upd, deployer=dep, max_retries=3)
    monkeypatch.setattr(subprocess, "run", _pytest_collection_failure)

    sleep_calls = []

    from menace import unified_update_service as uus

    monkeypatch.setattr(uus.time, "sleep", lambda sec: sleep_calls.append(sec))
    caplog.set_level("ERROR")

    with pytest.raises(subprocess.CalledProcessError):
        svc._cycle()

    assert sleep_calls == []
    assert "root-cause=import contract" in caplog.text
    assert "terminal test failure detected" in caplog.text


def test_retry_uses_exponential_backoff(monkeypatch):
    bus = UnifiedEventBus()
    upd = DummyUpdate()
    dep = DummyDeployer(bus)
    svc = UnifiedUpdateService(
        updater_service=upd,
        deployer=dep,
        max_retries=3,
        retry_backoff_seconds=0.5,
        retry_backoff_cap_seconds=1.2,
    )

    attempts = {"count": 0}

    def _always_fail(cmd, check=False):
        attempts["count"] += 1
        raise RuntimeError("network timeout")

    monkeypatch.setattr(subprocess, "run", _always_fail)
    sleep_calls = []

    from menace import unified_update_service as uus

    monkeypatch.setattr(uus.time, "sleep", lambda sec: sleep_calls.append(sec))

    with pytest.raises(RuntimeError):
        svc._cycle()

    assert attempts["count"] == 4
    assert sleep_calls == [0.5, 1.0, 1.2]
