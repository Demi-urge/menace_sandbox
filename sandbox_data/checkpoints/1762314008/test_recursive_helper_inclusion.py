import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def _setup(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("import helper\nimport redundant\n")  # path-ignore
    (repo / "helper.py").write_text(  # path-ignore
        "from pathlib import Path\nPath('helper_ran').write_text('x')\n"
    )
    (repo / "redundant.py").write_text(  # path-ignore
        "from pathlib import Path\nPath('redundant_ran').write_text('x')\n"
    )

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")
    monkeypatch.setenv("SANDBOX_TEST_REDUNDANT", "0")
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    gen = types.ModuleType("menace.environment_generator")
    gen._CPU_LIMITS = {}
    gen._MEMORY_LIMITS = {}
    monkeypatch.setitem(sys.modules, "menace.environment_generator", gen)

    od = types.ModuleType("sandbox_runner.orphan_discovery")
    od.discover_recursive_orphans = lambda path: {}
    od.discover_orphan_modules = lambda path: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_discovery", od)
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)

    mod = types.ModuleType("scripts.discover_isolated_modules")
    def discover(path, *, recursive=True):
        assert Path(path) == repo
        assert recursive is True
        return ["main.py", "helper.py", "redundant.py"]  # path-ignore
    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    import orphan_analyzer
    classified: dict[str, str] = {}
    def fake_classify(p):
        name = Path(p).name
        res = "redundant" if name == "redundant.py" else "candidate"  # path-ignore
        classified[name] = res
        return res
    monkeypatch.setattr(orphan_analyzer, "classify_module", fake_classify)
    import sandbox_settings as ss
    class DummySettings(ss.SandboxSettings):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.test_redundant_modules = False
    monkeypatch.setattr(ss, "SandboxSettings", DummySettings)

    svc_mod = types.ModuleType("self_test_service")
    class DummyService:
        def __init__(self, pytest_args, **kwargs):
            self.path = Path(pytest_args)
        def run_once(self):
            import runpy
            runpy.run_path(str(self.path), run_name="__main__")
            return {"failed": 0}
    svc_mod.SelfTestService = DummyService
    monkeypatch.setitem(sys.modules, "self_test_service", svc_mod)

    import sandbox_runner.environment as env
    env.generate_workflows_for_modules = (
        lambda mods, workflows_db="workflows.db", context_builder=None: None
    )
    env.try_integrate_into_workflows = (
        lambda mods, context_builder=None: None
    )
    env.run_workflow_simulations = (
        lambda: [types.SimpleNamespace(roi_history=[])]
    )

    return env, repo, data_dir, classified


def test_recursive_helper_execution(_setup):
    env, repo, data_dir, classified = _setup
    from context_builder_util import create_context_builder

    env.auto_include_modules(
        ["main.py"],
        recursive=True,
        validate=True,
        context_builder=create_context_builder(),
    )  # path-ignore

    assert (repo / "helper_ran").exists()
    data = json.loads((data_dir / "module_map.json").read_text())
    assert "helper.py" in data["modules"] and "redundant.py" not in data["modules"]  # path-ignore
    orphans = json.loads((data_dir / "orphan_modules.json").read_text())
    assert "redundant.py" in orphans and orphans["redundant.py"]["redundant"]  # path-ignore
    assert classified["redundant.py"] == "redundant"  # path-ignore
    assert classified["main.py"] == "candidate"  # path-ignore
