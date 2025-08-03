import types
import sys

from tests.test_recursive_orphans import _load_methods, DummyLogger


_, _, _, _test_orphan_modules = _load_methods()


def _make_engine():
    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)
    return eng


def test_orphan_module_positive_roi(monkeypatch, tmp_path):
    def fake_run(repo_path, modules=None, return_details=False, **k):
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0] for m in (modules or [])},
            metrics_history={"synergy_roi": [0.0]},
        )
        details = {m: {"sec": [{"result": {"exit_code": 0}}]} for m in (modules or [])}
        return tracker, details

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = fake_run
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    eng = _make_engine()
    assert eng._test_orphan_modules(["foo.py"]) == {"foo.py"}


def test_orphan_module_negative_roi(monkeypatch, tmp_path):
    def fake_run(repo_path, modules=None, return_details=False, **k):
        tracker = types.SimpleNamespace(
            module_deltas={m: [-1.0] for m in (modules or [])},
            metrics_history={"synergy_roi": [-0.5]},
        )
        details = {m: {"sec": [{"result": {"exit_code": 0}}]} for m in (modules or [])}
        return tracker, details

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = fake_run
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    eng = _make_engine()
    assert eng._test_orphan_modules(["foo.py"]) == set()

