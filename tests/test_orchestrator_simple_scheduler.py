import os
import sys
import types
import pytest

pytest.skip("optional dependencies not installed", allow_module_level=True)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sql_mod = types.ModuleType("sqlalchemy")
eng_mod = types.ModuleType("sqlalchemy.engine")
eng_mod.Engine = object
sql_mod.engine = eng_mod
sys.modules.setdefault("sqlalchemy", sql_mod)
sys.modules.setdefault("sqlalchemy.engine", eng_mod)
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
primitives = types.ModuleType("primitives")
sys.modules.setdefault("cryptography.hazmat.primitives", primitives)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
ed_mod = types.ModuleType("ed25519")
ed_mod.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed_mod.Ed25519PublicKey = object
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", ed_mod)
ser_mod = types.ModuleType("serialization")
primitives.serialization = ser_mod
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", ser_mod)
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
env_mod = types.ModuleType("env_config")
env_mod.DATABASE_URL = "sqlite:///:memory:"
env_mod.BUDGET_MAX_INSTANCES = 1
sys.modules.setdefault("env_config", env_mod)
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("requests", types.ModuleType("requests"))
git_mod = types.ModuleType("git")
git_mod.Repo = object
sys.modules.setdefault("git", git_mod)
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("pyplot"))  # path-ignore

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "menace.menace_orchestrator",
    Path(__file__).resolve().parents[1] / "menace_orchestrator.py",  # path-ignore
)
mo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mo)

class DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
    def start(self):
        pass
    def join(self, timeout=0):
        pass

def test_job_runs_and_shutdown(monkeypatch):
    monkeypatch.setattr(mo.threading, "Thread", DummyThread)
    monkeypatch.setattr(mo.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    sched = mo._SimpleScheduler()
    called = []
    def job():
        called.append(True)
    sched.add_job(job, interval=1, id="j")
    sched.tasks["j"] = (1, job, 0.0)
    with pytest.raises(SystemExit):
        sched._run()
    assert called == [True]
    sched.shutdown()


def test_remove_and_reschedule(monkeypatch):
    monkeypatch.setattr(mo.threading, "Thread", DummyThread)
    monkeypatch.setattr(mo.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    sched = mo._SimpleScheduler()
    called = []
    def job():
        called.append("run")
    sched.add_job(job, interval=1, id="j")
    sched.remove_job("j")
    with pytest.raises(SystemExit):
        sched._run()
    assert not called
    sched.add_job(job, interval=1, id="j")
    sched.reschedule_job("j", 0)
    sched.tasks["j"] = (0, job, 0.0)
    with pytest.raises(SystemExit):
        sched._run()
    assert called == ["run"]
    sched.shutdown()

def test_job_exception_logged(monkeypatch, caplog):
    monkeypatch.setattr(mo.threading, "Thread", DummyThread)
    monkeypatch.setattr(mo.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    sched = mo._SimpleScheduler()
    def boom():
        raise RuntimeError("x")
    sched.add_job(boom, interval=1, id="j")
    sched.tasks["j"] = (1, boom, 0.0)
    caplog.set_level("ERROR")
    with pytest.raises(SystemExit):
        sched._run()
    assert "job j failed" in caplog.text
    sched.shutdown()
