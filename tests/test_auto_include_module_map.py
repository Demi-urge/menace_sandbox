import sys
import types
import json
from pathlib import Path

import pytest

import sandbox_runner.environment as env


class DummyTracker:
    def save_history(self, path: str) -> None:
        Path(path).write_text(json.dumps({"roi_history": []}))


def test_auto_include_updates_module_map(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    def fake_generate(mods, workflows_db="workflows.db"):
        return [1]

    def fake_integrate(mods):
        return [1]

    tracker = DummyTracker()

    def fake_run():
        return tracker

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)
    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            analyze_redundancy=lambda path: False,
            classify_module=lambda path: "candidate",
        ),
    )

    env.auto_include_modules(["mod.py"])

    map_path = Path(tmp_path, "module_map.json")
    assert map_path.exists()
    data = json.loads(map_path.read_text())
    if "modules" in data:
        data = data["modules"]
    assert "mod.py" in data


@pytest.mark.parametrize(
    "existing",
    [
        {"mod.py": 1},
        {"modules": {"mod.py": 1}},
    ],
)
def test_auto_include_skips_existing(monkeypatch, tmp_path, existing):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    map_path = Path(tmp_path, "module_map.json")
    map_path.write_text(json.dumps(existing))

    called = {"gen": False, "int": False}

    def fake_generate(mods, workflows_db="workflows.db"):
        called["gen"] = True
        return []

    def fake_integrate(mods):
        called["int"] = True
        return []

    tracker = DummyTracker()

    def fake_run():
        return tracker

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)
    monkeypatch.setitem(
        sys.modules,
        "sandbox_settings",
        types.SimpleNamespace(
            SandboxSettings=lambda: types.SimpleNamespace(
                auto_include_isolated=False, recursive_isolated=False
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            analyze_redundancy=lambda path: False,
            classify_module=lambda path: "candidate",
        ),
    )
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")

    env.auto_include_modules(["mod.py"])

    assert json.loads(map_path.read_text()) == existing
    assert not called["gen"]
    assert not called["int"]


def test_redundant_module_validated_and_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    calls: dict[str, object] = {}

    def fake_generate(mods, workflows_db="workflows.db"):
        calls["generate"] = list(mods)

    def fake_integrate(mods):
        calls["integrate"] = list(mods)

    tracker = DummyTracker()

    def fake_run():
        calls["run"] = True
        return tracker

    class DummySelfTest:
        def __init__(self, pytest_args, **kw):
            calls["selftest"] = pytest_args

        def run_once(self):
            return {"failed": False}

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)
    monkeypatch.setitem(
        sys.modules,
        "self_test_service",
        types.SimpleNamespace(SelfTestService=DummySelfTest),
    )
    monkeypatch.setitem(
        sys.modules,
        "sandbox_settings",
        types.SimpleNamespace(
            SandboxSettings=lambda: types.SimpleNamespace(
                auto_include_isolated=False,
                recursive_isolated=False,
                test_redundant_modules=False,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            classify_module=lambda path: "redundant",
            analyze_redundancy=lambda path: True,
        ),
    )

    result, tested = env.auto_include_modules(["mod.py"], validate=True)

    assert result is tracker
    assert calls.get("selftest") == "mod.py"
    assert calls.get("run") is True
    assert "generate" not in calls and "integrate" not in calls
    assert tested == {"added": ["mod.py"], "failed": [], "redundant": ["mod.py"]}


def test_redundant_module_integrated_when_flag_set(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    calls: dict[str, object] = {}

    def fake_generate(mods, workflows_db="workflows.db"):
        calls["generate"] = list(mods)
        return []

    def fake_integrate(mods):
        calls["integrate"] = list(mods)
        return []

    tracker = DummyTracker()

    def fake_run():
        calls["run"] = True
        return tracker

    class DummySelfTest:
        def __init__(self, pytest_args, **kw):
            calls["selftest"] = pytest_args

        def run_once(self):
            return {"failed": False}

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)
    monkeypatch.setitem(
        sys.modules,
        "self_test_service",
        types.SimpleNamespace(SelfTestService=DummySelfTest),
    )
    monkeypatch.setitem(
        sys.modules,
        "sandbox_settings",
        types.SimpleNamespace(
            SandboxSettings=lambda: types.SimpleNamespace(
                auto_include_isolated=False,
                recursive_isolated=False,
                test_redundant_modules=True,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            classify_module=lambda path: "redundant",
            analyze_redundancy=lambda path: True,
        ),
    )

    result, tested = env.auto_include_modules(["mod.py"], validate=True)

    assert result is tracker
    assert calls.get("selftest") == "mod.py"
    assert calls.get("run") is True
    assert calls.get("generate") == ["mod.py"]
    assert calls.get("integrate") == ["mod.py"]
    map_path = Path(tmp_path, "module_map.json")
    assert map_path.exists()
    data = json.loads(map_path.read_text())
    if "modules" in data:
        data = data["modules"]
    assert "mod.py" in data
    assert tested == {"added": ["mod.py"], "failed": [], "redundant": ["mod.py"]}
