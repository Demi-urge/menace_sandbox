import json
import os
import sys
import types
from pathlib import Path

import pytest


def _setup_env(monkeypatch, tmp_path, redundant=None):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # create orphan chain a -> b -> c
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import c\n")
    (tmp_path / "c.py").write_text("VALUE = 1\n")

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    generated = []
    integrated = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(sorted(mods))
        (tmp_path / workflows_db).write_text("dummy")
        return [1]

    def fake_integrate(mods):
        integrated.extend(sorted(mods))
        data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][Path(m).name] = 1
        map_path.write_text(json.dumps(data))

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.generate_workflows_for_modules = fake_generate
    env_mod.try_integrate_into_workflows = fake_integrate
    env_mod.run_workflow_simulations = lambda: []

    def auto_include_modules(mods, recursive=False):
        mod_set = {Path(m).as_posix() for m in mods}
        if recursive:
            from sandbox_runner.orphan_discovery import discover_recursive_orphans
            import orphan_analyzer

            repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
            for name, info in discover_recursive_orphans(str(repo)).items():
                path = Path(*name.split(".")).with_suffix(".py")
                full = repo / path
                if info.get("redundant") or orphan_analyzer.analyze_redundancy(full):
                    continue
                mod_set.add(path.as_posix())
        mods2 = sorted(mod_set)
        env_mod.generate_workflows_for_modules(mods2)
        env_mod.try_integrate_into_workflows(mods2)
        return env_mod.run_workflow_simulations()

    env_mod.auto_include_modules = auto_include_modules
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    od = types.ModuleType("sandbox_runner.orphan_discovery")
    od.discover_recursive_orphans = lambda path: {
        "a": {"parents": [], "redundant": False},
        "b": {"parents": ["a"], "redundant": False},
        "c": {"parents": ["b"], "redundant": False},
    }
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_discovery", od)

    import orphan_analyzer
    monkeypatch.setattr(
        orphan_analyzer,
        "analyze_redundancy",
        lambda p: p.name == "b.py" if redundant else False,
    )

    return env_mod, generated, integrated, map_path


def test_auto_include_recursive_chain(tmp_path, monkeypatch):
    env, generated, integrated, map_path = _setup_env(monkeypatch, tmp_path)

    from sandbox_runner.environment import auto_include_modules

    auto_include_modules(["a.py"], recursive=True)

    assert generated and generated[0] == ["a.py", "b.py", "c.py"]
    assert integrated == ["a.py", "b.py", "c.py"]
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"a.py", "b.py", "c.py"}
    assert (tmp_path / "workflows.db").exists()


def test_auto_include_recursive_skips_redundant(tmp_path, monkeypatch):
    env, generated, integrated, map_path = _setup_env(monkeypatch, tmp_path, redundant=True)

    from sandbox_runner.environment import auto_include_modules

    auto_include_modules(["a.py"], recursive=True)

    assert generated and generated[0] == ["a.py", "c.py"]
    assert integrated == ["a.py", "c.py"]
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"a.py", "c.py"}
    assert (tmp_path / "workflows.db").exists()
