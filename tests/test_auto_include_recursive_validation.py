import json
import sys
import types
from pathlib import Path

import sandbox_runner.environment as env
from dynamic_path_router import resolve_path
from context_builder_util import create_context_builder


class DummyTracker:
    def save_history(self, path: str) -> None:
        Path(path).write_text(json.dumps({"roi_history": [0]}))


GOOD = "good.py"  # path-ignore
BAD = "bad.py"  # path-ignore


def test_recursive_skips_redundant(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    calls: dict[str, object] = {}

    (tmp_path / GOOD).write_text("VALUE = 1\n")
    (tmp_path / BAD).write_text("VALUE = 0\n")

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

    def classify(path):
        calls.setdefault("analyze", []).append(Path(path).name)
        return "redundant" if Path(path).name == BAD else "candidate"

    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            analyze_redundancy=lambda path: False,
            classify_module=classify,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.discover_isolated_modules",
        types.SimpleNamespace(discover_isolated_modules=lambda *a, **k: []),
    )

    good = resolve_path(GOOD).as_posix()
    bad = resolve_path(BAD).as_posix()
    result, tested = env.auto_include_modules(
        [good, bad], context_builder=create_context_builder()
    )

    assert result is tracker
    assert tested == {"added": [bad, good], "failed": [], "redundant": [bad]}
    assert calls["generate"] == [good]
    assert calls["integrate"] == [good]
    assert sorted(calls["analyze"]) == [BAD, GOOD]


def test_recursive_validated_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")
    calls: dict[str, object] = {}

    (tmp_path / GOOD).write_text("VALUE = 1\n")

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

    def classify(path):
        calls.setdefault("analyze", []).append(Path(path).name)
        return "candidate"

    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            analyze_redundancy=lambda path: False,
            classify_module=classify,
        ),
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

    good = resolve_path(GOOD).as_posix()
    result, tested = env.auto_include_modules(
        [good], validate=True, context_builder=create_context_builder()
    )

    assert result is tracker
    assert tested == {"added": [good], "failed": [], "redundant": []}
    assert calls["generate"] == [good]
    assert calls["integrate"] == [good]
    assert calls["selftest"] == [
        (
            good,
            {
                "include_orphans": False,
                "discover_orphans": False,
                "discover_isolated": True,
                "recursive_orphans": True,
                "recursive_isolated": True,
                "auto_include_isolated": True,
                "include_redundant": True,
                "disable_auto_integration": True,
            },
        )
    ]
    assert calls["analyze"].count(GOOD) == 1
    assert (tmp_path / "roi_history.json").exists()
