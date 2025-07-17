import importlib.util

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",
    submodule_search_locations=[str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import menace.error_bot as eb
import subprocess


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
    def fail(cmd, check=True):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(mod.subprocess, "run", fail)
    svc._run_once()
    cur = db.conn.execute("SELECT COUNT(*) FROM telemetry")
    assert cur.fetchone()[0] == 1

