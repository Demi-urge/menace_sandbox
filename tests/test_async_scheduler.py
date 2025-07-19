import asyncio
import importlib.util
from pathlib import Path
import os
import types
import pytest

ROOT = Path(__file__).resolve().parents[1]

def load_mod(name, file):
    import sys, types, importlib.machinery
    if 'menace' not in sys.modules:
        pkg = types.ModuleType('menace')
        pkg.__path__ = [str(ROOT)]
        pkg.__spec__ = importlib.machinery.ModuleSpec('menace', loader=None, is_package=True)
        sys.modules['menace'] = pkg
    spec = importlib.util.spec_from_file_location(f"menace.{name}", ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

cms = load_mod('cross_model_scheduler', 'cross_model_scheduler.py')
sts = load_mod('self_test_service', 'self_test_service.py')

class DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
    def start(self):
        pass
    def join(self, timeout=0):
        pass

def test_async_scheduler_run_and_shutdown(monkeypatch):
    monkeypatch.setattr(cms.threading, 'Thread', DummyThread)
    monkeypatch.setattr(cms.asyncio, 'sleep', lambda s: (_ for _ in ()).throw(SystemExit))
    sched = cms._AsyncScheduler()
    called = []
    def job():
        called.append(True)
    sched.add_job(job, interval=1, id='j')
    sched._next_runs['j'] = 0.0
    with pytest.raises(SystemExit):
        asyncio.run(sched._run())
    assert called == [True]
    sched.shutdown()


def test_self_test_async_records(monkeypatch):
    os.environ['USE_ASYNC_SCHEDULER'] = '1'
    monkeypatch.setattr(sts, 'BackgroundScheduler', None)
    recorded = {}
    class DummyScheduler:
        def add_job(self, func, interval, id):
            recorded['func'] = func
            recorded['interval'] = interval
            recorded['id'] = id
        def shutdown(self):
            pass
    monkeypatch.setattr(sts, '_AsyncScheduler', DummyScheduler)
    class DummyDB:
        def __init__(self):
            self.results = []
        def add_test_result(self, p, f):
            self.results.append((p, f))
    db = DummyDB()
    svc = sts.SelfTestService(db=db)
    monkeypatch.setattr(svc, '_run_once', lambda: svc.error_logger.db.add_test_result(1,0))
    svc.run_continuous(interval=5)
    assert recorded['id'] == 'self_test'
    recorded['func']()
    assert db.results == [(1,0)]
