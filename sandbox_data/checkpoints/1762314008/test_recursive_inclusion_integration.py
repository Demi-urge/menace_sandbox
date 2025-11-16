import json
import sys
import types
from pathlib import Path

import sandbox_runner.environment as env
from context_builder_util import create_context_builder


class DummyTracker:
    def save_history(self, path: str) -> None:
        Path(path).write_text("{}")


def test_recursive_inclusion_integration(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "iso.py").write_text("import helper\nimport bad\n")  # path-ignore
    (repo / "helper.py").write_text("x = 1\n")  # path-ignore
    (repo / "bad.py").write_text("x = 2\n")  # path-ignore

    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    class Settings:
        auto_include_isolated = True
        recursive_isolated = True
        recursive_orphan_scan = True

    monkeypatch.setitem(
        sys.modules,
        "sandbox_settings",
        types.SimpleNamespace(SandboxSettings=lambda: Settings()),
    )

    def collect_deps(mods):
        mods = set(mods)
        if "iso.py" in mods:  # path-ignore
            mods.update({"helper.py", "bad.py"})  # path-ignore
        return mods

    monkeypatch.setitem(
        sys.modules,
        "sandbox_runner.dependency_utils",
        types.SimpleNamespace(collect_local_dependencies=collect_deps),
    )

    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            analyze_redundancy=lambda p: Path(p).name == "bad.py",  # path-ignore
            classify_module=lambda p: "redundant"
            if Path(p).name == "bad.py"  # path-ignore
            else "candidate",
        ),
    )

    iso_mod = types.SimpleNamespace(
        discover_isolated_modules=lambda repo, recursive=True: ["iso.py"]  # path-ignore
    )
    scripts_pkg = types.SimpleNamespace(discover_isolated_modules=iso_mod)
    monkeypatch.setitem(sys.modules, "scripts", scripts_pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", iso_mod)

    class STS:
        def __init__(self, pytest_args, **kwargs):
            pass

        def run_once(self):
            return ({}, [])

    monkeypatch.setitem(
        sys.modules, "self_test_service", types.SimpleNamespace(SelfTestService=STS)
    )

    monkeypatch.setattr(env, "generate_workflows_for_modules", lambda mods: None)
    monkeypatch.setattr(env, "try_integrate_into_workflows", lambda mods: None)
    monkeypatch.setattr(env, "run_workflow_simulations", lambda *a, **k: DummyTracker())

    env.auto_include_modules(
        ["iso.py"], recursive=True, validate=True, context_builder=create_context_builder()
    )  # path-ignore

    map_data = json.loads((data_dir / "module_map.json").read_text())
    assert set(map_data) == {"iso.py", "helper.py"}  # path-ignore
    assert "bad.py" not in map_data  # path-ignore

    orphan_cache = json.loads((data_dir / "orphan_modules.json").read_text())
    assert orphan_cache.get("bad.py", {}).get("redundant") is True  # path-ignore
