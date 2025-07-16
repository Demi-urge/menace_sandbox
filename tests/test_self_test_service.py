import importlib.util

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",
    submodule_search_locations=[str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


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

