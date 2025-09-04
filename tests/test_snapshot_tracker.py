import json
import importlib
import sys
import types
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
pkg_path = root_path / "self_improvement"
root_pkg = types.ModuleType("menace_sandbox")
root_pkg.__path__ = [str(root_path)]
sys.modules["menace_sandbox"] = root_pkg
package = types.ModuleType("menace_sandbox.self_improvement")
package.__path__ = [str(pkg_path)]
sys.modules["menace_sandbox.self_improvement"] = package

sys.modules["codebase_diff_checker"] = types.ModuleType("codebase_diff_checker")
sys.modules["logging_utils"] = types.SimpleNamespace(log_record=lambda **k: {})
stub_pm = types.ModuleType("prompt_memory")
stub_pm.log_prompt_attempt = lambda *a, **k: None
sys.modules["menace_sandbox.self_improvement.prompt_memory"] = stub_pm
sys.modules["menace_sandbox.self_improvement.metrics"] = types.ModuleType("metrics")
sys.modules["menace_sandbox.self_improvement.module_graph_analyzer"] = types.ModuleType(
    "module_graph_analyzer"
)
dyn = types.ModuleType("dynamic_path_router")
dyn.resolve_path = lambda p: p
sys.modules["dynamic_path_router"] = dyn
ss_mod = types.ModuleType("sandbox_settings")


class _SandboxSettings:
    sandbox_data_dir = "."
    sandbox_score_db = ""


ss_mod.SandboxSettings = _SandboxSettings
sys.modules["sandbox_settings"] = ss_mod
sys.modules["menace_sandbox.sandbox_settings"] = ss_mod
BaselineTracker = importlib.import_module(
    "menace_sandbox.self_improvement.baseline_tracker"
).BaselineTracker
ss = importlib.import_module("menace_sandbox.self_improvement.state_snapshot")


def test_checkpoint_and_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "resolve_path", lambda p: p)
    repo = tmp_path / ss.resolve_path("repo")
    repo.mkdir()
    module = repo / ss.resolve_path("module.py")
    module.write_text("a = 1\n", encoding="utf-8")

    data_dir = tmp_path / ss.resolve_path("data")

    class SettingsStub:
        sandbox_data_dir = str(data_dir)

    monkeypatch.setattr(ss, "SandboxSettings", lambda: SettingsStub())

    tracker = ss.SnapshotTracker(repo, BaselineTracker())
    tracker.last_snapshot = ss.StateSnapshot(1, 1, 1, 1, 1)

    after = ss.StateSnapshot(2, 2, 2, 2, 2)

    diff = repo / ss.resolve_path("change.diff")
    diff.write_text(
        f"""diff --git a/{module.name} b/{module.name}
--- a/{module.name}
+++ b/{module.name}
@@
-a = 1
+a = 2
""",
        encoding="utf-8",
    )

    # simulate modified file
    module.write_text("a = 2\n", encoding="utf-8")

    tracker.evaluate_change(after, {"strategy": "alpha"}, diff)

    ckpt_base = data_dir / ss.resolve_path("checkpoints")
    dirs = list(ckpt_base.iterdir())
    assert len(dirs) == 1
    assert (dirs[0] / module.name).exists()

    conf = json.loads((data_dir / ss.resolve_path("strategy_confidence.json")).read_text())
    assert conf["alpha"] == 1
