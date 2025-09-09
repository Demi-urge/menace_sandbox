import asyncio
import importlib.util
from pathlib import Path
import os
import types
import pytest

from dynamic_path_router import resolve_path

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

cms = load_mod('cross_model_scheduler', resolve_path('cross_model_scheduler.py'))  # path-ignore
sts = load_mod('self_test_service', resolve_path('self_test_service.py'))  # path-ignore


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}

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
    sched._next_runs['j'] = cms.time.time()
    with pytest.raises(SystemExit):
        asyncio.run(sched._run())
    assert called == [True]
    sched.shutdown()


def test_self_test_async_records(monkeypatch):
    class DummyDB:
        def __init__(self):
            self.results = []
        def add_test_result(self, p, f):
            self.results.append((p, f))

    db = DummyDB()
    svc = sts.SelfTestService(db=db, context_builder=DummyBuilder())

    async def fake_run_once():
        db.add_test_result(1, 0)

    monkeypatch.setattr(svc, '_run_once', fake_run_once)

    async def runner():
        loop = asyncio.get_running_loop()
        svc.run_continuous(interval=0.01, loop=loop)
        await asyncio.sleep(0.03)
        await svc.stop()

    asyncio.run(runner())
    assert db.results
