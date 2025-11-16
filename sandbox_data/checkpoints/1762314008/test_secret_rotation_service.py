import importlib.util
import types
import sys

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

# Load module dynamically to avoid importing entire package
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg
spec = importlib.util.spec_from_file_location(
    "menace.secret_rotation_service",
    ROOT / "secret_rotation_service.py",  # path-ignore
    submodule_search_locations=[str(ROOT)],
)
svc_mod = importlib.util.module_from_spec(spec)
sys.modules["menace.secret_rotation_service"] = svc_mod
spec.loader.exec_module(svc_mod)


class DummyManager:
    def __init__(self):
        self.calls = []

    def get(self, name, rotate=True):
        self.calls.append((name, rotate))


def test_scheduler_adds_job(monkeypatch):
    mgr = DummyManager()
    svc = svc_mod.SecretRotationService(manager=mgr, names=['a'])
    monkeypatch.setattr(svc_mod, 'BackgroundScheduler', None)
    recorded = {}

    def fake_add_job(self, func, interval, id):
        recorded['func'] = func
        recorded['interval'] = interval
        recorded['id'] = id

    monkeypatch.setattr(svc_mod._SimpleScheduler, 'add_job', fake_add_job)
    svc.run_continuous(interval=123)
    assert recorded['interval'] == 123
    assert recorded['id'] == 'secret_rotation'
    recorded['func']()
    assert mgr.calls == [('a', True)]


def test_run_once_rotates():
    mgr = DummyManager()
    svc = svc_mod.SecretRotationService(manager=mgr, names=['x', 'y'])
    svc.run_once()
    assert ('x', True) in mgr.calls
    assert ('y', True) in mgr.calls
