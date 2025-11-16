import json
import os
from pathlib import Path
import types

import pytest


def _setup_env(monkeypatch, tmp_path, redundant=None):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # create orphan chain a -> b -> c
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\n")  # path-ignore
    (tmp_path / "c.py").write_text("VALUE = 1\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    generated: list[list[str]] = []
    integrated: list[str] = []

    # Stub out workflow helpers to capture inputs
    def fake_generate(mods, workflows_db="workflows.db", context_builder=None):
        generated.append(sorted(mods))
        (tmp_path / workflows_db).write_text("dummy")
        return [1]

    def fake_integrate(mods, context_builder=None):
        integrated.extend(sorted(mods))
        data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][Path(m).name] = 1
        map_path.write_text(json.dumps(data))

    import sandbox_runner.environment as env_mod

    monkeypatch.setattr(env_mod, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env_mod, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env_mod, "run_workflow_simulations", lambda: [])

    import orphan_analyzer

    monkeypatch.setattr(
        orphan_analyzer,
        "analyze_redundancy",
        lambda p: p.name == "b.py" if redundant else False,  # path-ignore
    )

    return env_mod, generated, integrated, map_path


def test_auto_include_recursive_chain(tmp_path, monkeypatch):
    env, generated, integrated, map_path = _setup_env(monkeypatch, tmp_path)

    from sandbox_runner.environment import auto_include_modules
    from sandbox_runner.orphan_discovery import discover_recursive_orphans

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["c"]["parents"] == ["b"]

    ctx = types.SimpleNamespace(orphan_traces={})
    from context_builder_util import create_context_builder

    auto_include_modules(["a.py"], recursive=True, context_builder=create_context_builder())  # path-ignore

    assert generated and generated[0] == ["a.py", "b.py", "c.py"]  # path-ignore
    assert integrated == ["a.py", "b.py", "c.py"]  # path-ignore
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"a.py", "b.py", "c.py"}  # path-ignore
    assert ctx.orphan_traces["b.py"]["parents"] == ["a.py"]  # path-ignore
    assert ctx.orphan_traces["c.py"]["parents"] == ["b.py", "a.py"]  # path-ignore
    assert (tmp_path / "workflows.db").exists()


def test_auto_include_recursive_skips_redundant(tmp_path, monkeypatch):
    env, generated, integrated, map_path = _setup_env(monkeypatch, tmp_path, redundant=True)

    from sandbox_runner.environment import auto_include_modules
    from sandbox_runner.orphan_discovery import discover_recursive_orphans

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["c"]["parents"] == ["b"]

    ctx = types.SimpleNamespace(orphan_traces={})
    from context_builder_util import create_context_builder

    auto_include_modules(["a.py"], recursive=True, context_builder=create_context_builder())  # path-ignore

    assert ctx.orphan_traces["b.py"]["parents"] == ["a.py"]  # path-ignore
    assert ctx.orphan_traces["c.py"]["parents"] == ["b.py", "a.py"]  # path-ignore
