import asyncio
import importlib.util
import importlib.machinery
import types
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_self_test_service():
    if 'menace' in sys.modules:
        del sys.modules['menace']
    pkg = types.ModuleType('menace')
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec('menace', loader=None, is_package=True)
    sys.modules['menace'] = pkg
    spec = importlib.util.spec_from_file_location('menace.self_test_service', ROOT / 'self_test_service.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['menace.self_test_service'] = mod
    spec.loader.exec_module(mod)
    return mod


sts = load_self_test_service()


def test_async_loop(monkeypatch):
    class DummyDB:
        def __init__(self):
            self.results = []
        def add_test_result(self, p, f):
            self.results.append((p, f))

    db = DummyDB()
    svc = sts.SelfTestService(db=db)

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
