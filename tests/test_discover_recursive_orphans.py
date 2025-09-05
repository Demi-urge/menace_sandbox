import importlib
import json
import sys
import types


def _import_recursive(monkeypatch):
    env = types.ModuleType("env")
    env.simulate_execution_environment = lambda *_a, **_k: None
    env.generate_sandbox_report = lambda *_a, **_k: None
    env.run_repo_section_simulations = lambda *_a, **_k: None
    env.run_workflow_simulations = lambda *_a, **_k: None
    env.simulate_full_environment = lambda *_a, **_k: None
    env.generate_input_stubs = lambda *_a, **_k: None
    env.SANDBOX_INPUT_STUBS = {}
    env.SANDBOX_EXTRA_METRICS = {}
    env.SANDBOX_ENV_PRESETS = {}
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env)
    men = types.ModuleType("menace")
    env_gen = types.ModuleType("menace.environment_generator")
    env_gen._CPU_LIMITS = [1]
    env_gen._MEMORY_LIMITS = [1]
    men.environment_generator = env_gen
    monkeypatch.setitem(sys.modules, "menace", men)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_gen)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    mod = importlib.reload(importlib.import_module("sandbox_runner"))
    return mod.discover_recursive_orphans


def test_recursive_import_includes_dependencies(tmp_path, monkeypatch):
    discover_recursive_orphans = _import_recursive(monkeypatch)
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("x = 1\n")  # path-ignore
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"modules": {}}))

    res = discover_recursive_orphans(
        str(tmp_path), module_map=data_dir / "module_map.json"
    )
    assert set(res.keys()) == {"a", "b"}
    assert res["a"]["classification"] == "candidate"
    assert res["b"]["classification"] == "candidate"
    assert res["b"]["parents"] == ["a"]
    assert not res["a"]["redundant"]
    assert not res["b"]["redundant"]


def test_public_import(monkeypatch):
    func = _import_recursive(monkeypatch)
    assert callable(func)
    assert "sandbox_runner.sandbox_runner" not in sys.modules

import orphan_analyzer
from sandbox_runner.orphan_discovery import discover_recursive_orphans as _discover_inner


def test_discover_orphans_marks_redundant(tmp_path, monkeypatch):
    (tmp_path / "dup.py").write_text("pass\n")  # path-ignore
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"modules": {}}))

    def fake_classify(p, include_meta=False):
        if include_meta:
            return "redundant", {}
        return "redundant"

    monkeypatch.setattr(orphan_analyzer, "classify_module", fake_classify)

    res = _discover_inner(str(tmp_path), module_map=data_dir / "module_map.json")
    assert list(res.keys()) == ["dup"]
    info = res["dup"]
    assert info["classification"] == "redundant"
    assert info["redundant"] is True


def test_skip_dirs_env(tmp_path, monkeypatch):
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "ignore.py").write_text("pass\n")  # path-ignore
    (tmp_path / "keep.py").write_text("pass\n")  # path-ignore
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"modules": {}}))

    monkeypatch.setenv("SANDBOX_SKIP_DIRS", "foo")

    res = _discover_inner(str(tmp_path), module_map=data_dir / "module_map.json")
    assert "foo.ignore" not in res
    assert "keep" in res
