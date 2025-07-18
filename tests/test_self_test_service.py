import importlib.util

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",
)
mod = importlib.util.module_from_spec(spec)
import sys
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
spec.loader.exec_module(mod)
import menace.error_bot as eb
import subprocess
import types


def test_scheduler_start(monkeypatch):
    monkeypatch.setattr(mod, 'BackgroundScheduler', None)
    recorded = {}

    def fake_add_job(self, func, interval, id):
        recorded['id'] = id
        recorded['func'] = func

    monkeypatch.setattr(mod._SimpleScheduler, 'add_job', fake_add_job)
    svc = mod.SelfTestService()
    svc.run_continuous(interval=10)
    assert recorded['id'] == 'self_test'


def test_failure_logs_telemetry(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e.db")
    svc = mod.SelfTestService(db)
    def fail(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 1, stdout="1 failed, 0 passed", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fail)
    svc._run_once()
    cur = db.conn.execute("SELECT COUNT(*) FROM telemetry")
    assert cur.fetchone()[0] == 1
    row = db.conn.execute("SELECT passed, failed FROM test_results").fetchone()
    assert row == (0, 1)


def test_success_logs_results(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e2.db")
    svc = mod.SelfTestService(db)

    def succeed(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 0, stdout="3 passed in 0.1s", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", succeed)
    svc._run_once()
    cur = db.conn.execute("SELECT COUNT(*) FROM telemetry")
    assert cur.fetchone()[0] == 0
    row = db.conn.execute("SELECT passed, failed FROM test_results").fetchone()
    assert row == (3, 0)


def test_custom_args(monkeypatch):
    recorded = {}

    def fake_run(cmd, capture_output=True, text=True):
        recorded['cmd'] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="0 passed", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    svc = mod.SelfTestService(pytest_args="-k pattern")
    svc._run_once()
    assert "-k" in recorded['cmd'] and "pattern" in recorded['cmd']

