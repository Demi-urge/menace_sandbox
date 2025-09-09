import json
import sys
import types
from pathlib import Path

import sandbox_runner.environment as env
from dynamic_path_router import resolve_path
from context_builder_util import create_context_builder


class DummyTracker:
    def save_history(self, path: str) -> None:
        Path(path).write_text(json.dumps({"roi_history": [1.0]}))


def test_auto_include_modules_saves_roi(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")
    calls = {}

    def fake_generate(mods, workflows_db="workflows.db"):
        calls["generate"] = list(mods)
        return [1]

    def fake_integrate(mods):
        calls["integrate"] = list(mods)
        return [1]

    tracker = DummyTracker()

    def fake_run():
        calls["run"] = True
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

    MOD = resolve_path("mod.py").as_posix()  # path-ignore
    result, tested = env.auto_include_modules(
        [MOD], context_builder=create_context_builder()
    )

    assert result is tracker
    assert tested == {"added": [MOD], "failed": [], "redundant": []}
    assert calls.get("run") is True
    history = Path(tmp_path, "roi_history.json")
    assert history.exists()
    data = json.loads(history.read_text())
    assert data["roi_history"] == [1.0]
