import json
import sys
import types
from pathlib import Path

import pytest
from context_builder_util import create_context_builder


def _setup(monkeypatch, tmp_path, *, redundant=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("import dep\n")  # path-ignore
    (repo / "dep.py").write_text("x = 1\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    import sandbox_runner.orphan_discovery as od
    monkeypatch.setattr(od, "discover_recursive_orphans", lambda path: {})

    mod = types.ModuleType("scripts.discover_isolated_modules")
    def discover(path, *, recursive=True):
        assert Path(path) == repo
        assert recursive is True
        return ["iso.py", "dep.py"]  # path-ignore
    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    import orphan_analyzer
    calls: dict[str, int] = {}
    def fake_analyze(path: Path) -> bool:
        name = path.name
        count = calls.get(name, 0)
        calls[name] = count + 1
        if redundant and name == "dep.py":  # path-ignore
            return count > 0
        return False
    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", fake_analyze)

    svc_mod = types.ModuleType("self_test_service")
    class DummyService:
        def __init__(self, *a, **k):
            pass
        def run_once(self):
            return {"failed": 0}
    svc_mod.SelfTestService = DummyService
    monkeypatch.setitem(sys.modules, "self_test_service", svc_mod)

    import sandbox_runner.environment as env
    generated: list[list[str]] = []
    integrated: list[str] = []
    def fake_generate(mods, workflows_db="workflows.db", context_builder=None):
        generated.append(sorted(mods))
        return [1]
    def fake_integrate(mods, context_builder=None):
        integrated.extend(sorted(mods))
        data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][Path(m).name] = 1
        map_path.write_text(json.dumps(data))
    env.generate_workflows_for_modules = fake_generate
    env.try_integrate_into_workflows = fake_integrate
    env.run_workflow_simulations = lambda: [types.SimpleNamespace(roi_history=[])]

    return env, map_path, data_dir, generated, integrated


def test_auto_include_isolated_dependency(tmp_path, monkeypatch):
    env, map_path, data_dir, generated, integrated = _setup(monkeypatch, tmp_path)
    env.auto_include_modules(
        ["iso.py"], recursive=True, validate=True, context_builder=create_context_builder()
    )  # path-ignore

    assert generated and generated[0] == ["dep.py", "iso.py"]  # path-ignore
    assert integrated == ["dep.py", "iso.py"]  # path-ignore
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"iso.py", "dep.py"}  # path-ignore
    orphan_file = data_dir / "orphan_modules.json"
    if orphan_file.exists():
        assert json.loads(orphan_file.read_text()) == {}


def test_auto_include_isolated_skips_redundant(tmp_path, monkeypatch):
    env, map_path, data_dir, generated, integrated = _setup(monkeypatch, tmp_path, redundant=True)
    env.auto_include_modules(
        ["iso.py"], recursive=True, validate=True, context_builder=create_context_builder()
    )  # path-ignore

    assert generated and generated[0] == ["iso.py"]  # path-ignore
    assert integrated == ["iso.py"]  # path-ignore
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"iso.py"}  # path-ignore
    orphan_file = data_dir / "orphan_modules.json"
    info = json.loads(orphan_file.read_text())
    assert info == {"dep.py": {"redundant": True}}  # path-ignore
