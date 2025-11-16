import importlib.util
import sqlite3
import types
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

class DummyUpdater:
    def __init__(self):
        self.calls = 0
    def run_cycle(self, update_os=False):
        self.calls += 1
        return ["pkg"]

class DummyDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE update_history(packages TEXT, status TEXT, ts TEXT)"
        )
    def add_update(self, packages, status):
        self.conn.execute(
            "INSERT INTO update_history(packages, status, ts) VALUES (?,?,?)",
            (";".join(packages), status, "t"),
        )
        self.conn.commit()

def _load_service():
    stub = types.ModuleType("menace.deployment_bot")
    stub.DeploymentDB = DummyDB
    sys.modules["menace.deployment_bot"] = stub
    spec = importlib.util.spec_from_file_location(
        "menace.dependency_update_service",
        ROOT / "dependency_update_service.py",  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["menace.dependency_update_service"] = mod
    spec.loader.exec_module(mod)
    return mod

def test_scheduler_adds_job(monkeypatch):
    service_mod = _load_service()
    updater = DummyUpdater()
    svc = service_mod.DependencyUpdateService(updater=updater, db=DummyDB())
    monkeypatch.setattr(service_mod, "BackgroundScheduler", None)
    recorded = {}
    def fake_add_job(self, func, interval, id):
        recorded["func"] = func
        recorded["interval"] = interval
        recorded["id"] = id
    monkeypatch.setattr(service_mod._SimpleScheduler, "add_job", fake_add_job)
    svc.run_continuous(interval=123)
    assert recorded["interval"] == 123
    assert recorded["id"] == "dep_update"
    recorded["func"]()
    assert updater.calls == 1


def test_run_once_records(monkeypatch):
    service_mod = _load_service()
    db = DummyDB()
    updater = DummyUpdater()
    svc = service_mod.DependencyUpdateService(updater=updater, db=db)
    def fake_run(cmd, check=False):
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    svc._run_once()
    row = db.conn.execute("SELECT status FROM update_history").fetchone()
    assert row[0] == "success"


def test_run_once_remote_verify(monkeypatch):
    service_mod = _load_service()
    db = DummyDB()
    updater = DummyUpdater()
    svc = service_mod.DependencyUpdateService(updater=updater, db=db)
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    svc._run_once(verify_host="host")
    assert [
        "ssh",
        "host",
        "docker",
        "run",
        "--rm",
        "menace:latest",
        "pytest",
        "-q",
    ] in calls

