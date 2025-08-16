import types
import importlib
import atexit
from pathlib import Path

from tests.test_recursive_orphans import _load_methods, DummyLogger


_, _, _, _test_orphan_modules = _load_methods()


def test_retire_flag_skips_module(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    eng = types.SimpleNamespace(logger=DummyLogger())
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)

    rr = importlib.import_module("relevancy_radar")
    monkeypatch.setattr(atexit, "register", lambda func: func)
    monkeypatch.setattr(rr, "_MODULE_USAGE_FILE", tmp_path / "module_usage.json")
    monkeypatch.setattr(rr, "_RELEVANCY_FLAGS_FILE", tmp_path / "relevancy_flags.json")
    rr._module_usage_counter.clear()
    rr._relevancy_flags.clear()
    monkeypatch.setattr(rr, "load_usage_stats", lambda: {"other.py": 1})

    res = eng._test_orphan_modules(["foo.py"])
    assert res == set()
    assert eng.orphan_traces["foo.py"]["radar_flag"] == "retire"
