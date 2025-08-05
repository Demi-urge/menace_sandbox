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
    calls: dict[str, object] = {}

    (tmp_path / "good.py").write_text("VALUE = 1\n")
    (tmp_path / "bad.py").write_text("VALUE = 0\n")

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

    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(analyze_redundancy=analyze),
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.discover_isolated_modules",
        types.SimpleNamespace(discover_isolated_modules=lambda *a, **k: []),
    )

    result, tested = env.auto_include_modules(["good.py", "bad.py"])

    assert result is tracker
    assert tested == {"added": ["good.py"], "failed": [], "redundant": ["bad.py"]}
    assert calls["generate"] == ["good.py"]
    assert calls["integrate"] == ["good.py"]
    assert sorted(calls["analyze"]) == ["bad.py", "good.py"]


def test_recursive_validated_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    calls: dict[str, object] = {}

    (tmp_path / "good.py").write_text("VALUE = 1\n")

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

    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(analyze_redundancy=analyze),
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.discover_isolated_modules",
        types.SimpleNamespace(discover_isolated_modules=lambda *a, **k: []),
    )

    class DummySelfTest:
        def __init__(self, pytest_args, **kwargs):
            calls.setdefault("selftest", []).append((pytest_args, kwargs))

        def run_once(self):
            return {"failed": False}

    monkeypatch.setitem(
        sys.modules,
        "self_test_service",
        types.SimpleNamespace(SelfTestService=DummySelfTest),
    )

    result, tested = env.auto_include_modules(["good.py"], validate=True)

    assert result is tracker
    assert tested == {"added": ["good.py"], "failed": [], "redundant": []}
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
                "recursive_isolated": True,
                "auto_include_isolated": True,
                "disable_auto_integration": True,
            },
        )
    ]
    assert calls["analyze"].count("good.py") == 1
    assert (tmp_path / "roi_history.json").exists()
