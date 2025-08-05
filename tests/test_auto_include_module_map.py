import sys
import types
import json
from pathlib import Path

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
        types.SimpleNamespace(analyze_redundancy=lambda path: False),
    )

    env.auto_include_modules(["mod.py"])

    map_path = Path(tmp_path, "module_map.json")
    assert map_path.exists()
    data = json.loads(map_path.read_text())
    if "modules" in data:
        data = data["modules"]
    assert "mod.py" in data
