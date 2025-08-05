import json
import sys
import types
from pathlib import Path

import sandbox_runner.environment as env


class DummyTracker:
    def save_history(self, path: str) -> None:
        Path(path).write_text(json.dumps({"roi_history": [0]}))


def test_recursive_skips_redundant(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "1")
    calls: dict[str, object] = {}

    def fake_discover(repo_path: str):
        calls["discover"] = repo_path
        return {"good": {"parents": []}, "bad": {"parents": []}}

    monkeypatch.setattr(
        "sandbox_runner.orphan_discovery.discover_recursive_orphans", fake_discover
    )

    def fake_generate(mods, workflows_db="workflows.db"):
        calls["generate"] = list(mods)

    def fake_integrate(mods):
        calls["integrate"] = list(mods)

    tracker = DummyTracker()

    def fake_run():
        calls["run"] = True
        return tracker

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)

    def analyze(path):
        calls.setdefault("analyze", []).append(Path(path).name)
        return Path(path).name == "bad.py"

    monkeypatch.setitem(sys.modules, "orphan_analyzer", types.SimpleNamespace(analyze_redundancy=analyze))

    result, tested = env.auto_include_modules([])

    assert result is tracker
    assert tested == {"added": ["good.py"], "failed": [], "redundant": ["bad.py"]}
    assert calls["discover"] == str(tmp_path)
    assert calls["generate"] == ["good.py"]
    assert calls["integrate"] == ["good.py"]
    assert sorted(calls["analyze"]) == ["bad.py", "good.py", "good.py"]


def test_recursive_validated_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "1")
    calls: dict[str, object] = {}

    def fake_discover(repo_path: str):
        calls["discover"] = repo_path
        return {"good": {"parents": []}}

    monkeypatch.setattr(
        "sandbox_runner.orphan_discovery.discover_recursive_orphans", fake_discover
    )

    def fake_generate(mods, workflows_db="workflows.db"):
        calls["generate"] = list(mods)

    def fake_integrate(mods):
        calls["integrate"] = list(mods)

    tracker = DummyTracker()

    def fake_run():
        calls["run"] = True
        return tracker

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)

    def analyze(path):
        calls.setdefault("analyze", []).append(Path(path).name)
        return False

    monkeypatch.setitem(sys.modules, "orphan_analyzer", types.SimpleNamespace(analyze_redundancy=analyze))

    class DummySelfTest:
        def __init__(self, pytest_args, **kwargs):
            calls.setdefault("selftest", []).append((pytest_args, kwargs))

        def run_once(self):
            return {"failed": False}

    monkeypatch.setitem(
        sys.modules, "self_test_service", types.SimpleNamespace(SelfTestService=DummySelfTest)
    )

    result, tested = env.auto_include_modules([], validate=True)

    assert result is tracker
    assert tested == {"added": ["good.py"], "failed": [], "redundant": []}
    assert calls["discover"] == str(tmp_path)
    assert calls["generate"] == ["good.py"]
    assert calls["integrate"] == ["good.py"]
    assert calls["selftest"] == [
        (
            "good.py",
            {
                "include_orphans": False,
                "discover_orphans": False,
                "discover_isolated": True,
                "recursive_orphans": True,
                "disable_auto_integration": True,
            },
        )
    ]
    assert calls["analyze"].count("good.py") == 2
    assert (tmp_path / "roi_history.json").exists()
