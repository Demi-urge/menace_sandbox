import json
import sys
import types
from pathlib import Path

import sandbox_runner.environment as env
from context_builder_util import create_context_builder


class DummyTracker:
    def __init__(self, roi):
        self.roi_history = [roi]
        self.cluster_map = {}

    def save_history(self, path: str) -> None:
        Path(path).write_text(json.dumps({"roi_history": self.roi_history}))


def test_auto_include_reverts_on_negative_roi(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")

    def fake_generate(mods, workflows_db="workflows.db"):
        return [1]

    def fake_integrate(mods):
        return [1]

    calls = {"run": 0}
    tracker1 = DummyTracker(1.0)
    tracker2 = DummyTracker(0.5)

    def fake_run():
        calls["run"] += 1
        return tracker1 if calls["run"] == 1 else tracker2

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", fake_run)
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
            analyze_redundancy=lambda path: False,
            classify_module=lambda path: "candidate",
        ),
    )

    result, tested = env.auto_include_modules(
        ["mod.py"], context_builder=create_context_builder()
    )  # path-ignore

    map_path = Path(tmp_path, "module_map.json")
    assert not map_path.exists()

    cache_path = Path(tmp_path, "orphan_modules.json")
    data = json.loads(cache_path.read_text())
    assert data["mod.py"]["rejected"] is True  # path-ignore
    assert tested == {"added": [], "failed": [], "redundant": [], "rejected": ["mod.py"]}  # path-ignore
    assert result is tracker1
    history = json.loads(Path(tmp_path, "roi_history.json").read_text())
    assert history["roi_history"] == [1.0]
