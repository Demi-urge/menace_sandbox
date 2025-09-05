import types
import importlib
import atexit
import sys

from tests.test_recursive_orphans import _load_methods, DummyLogger


_, _, _, _test_orphan_modules = _load_methods()


def test_retire_flag_skips_module(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)

    # Track calls to sandbox_runner to ensure retired modules are not executed
    calls: list[list[str]] = []
    sr = types.ModuleType("sandbox_runner")

    def fake_run_repo_section_simulations(repo_path, modules=None, **_k):
        calls.append(list(modules or []))
        tracker = types.SimpleNamespace(roi_history=[], metrics_history={"synergy_roi": []})
        return tracker, {}

    sr.run_repo_section_simulations = fake_run_repo_section_simulations
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    rr = importlib.import_module("relevancy_radar")
    monkeypatch.setattr(atexit, "register", lambda func: func)
    monkeypatch.setattr(rr, "_MODULE_USAGE_FILE", tmp_path / "module_usage.json")
    monkeypatch.setattr(rr, "_RELEVANCY_FLAGS_FILE", tmp_path / "relevancy_flags.json")
    rr._module_usage_counter.clear()
    rr._relevancy_flags.clear()
    monkeypatch.setattr(rr, "load_usage_stats", lambda: {"other.py": 1})  # path-ignore

    res = eng._test_orphan_modules(["foo.py"])  # path-ignore
    assert res == set()
    assert eng.orphan_traces["foo.py"]["radar_flag"] == "retire"  # path-ignore
    assert calls == []
