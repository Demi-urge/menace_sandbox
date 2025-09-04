import importlib
import json
import sys
import types
from pathlib import Path

def test_confidence_and_best_checkpoint(tmp_path, monkeypatch):
    stub = types.SimpleNamespace(
        collect_snapshot_metrics=lambda *a, **k: (0.0, 0.0),
        compute_call_graph_complexity=lambda *a, **k: 0.0,
    )
    sys.modules["menace_sandbox.self_improvement.metrics"] = stub
    sys.modules["dynamic_path_router"] = types.SimpleNamespace(
        resolve_path=lambda p: p,
        resolve_dir=lambda p: Path(p),
    )
    st = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
    monkeypatch.setattr(st, "resolve_path", lambda p: p)

    class Settings:
        sandbox_data_dir = str(tmp_path)

    monkeypatch.setattr(st, "SandboxSettings", lambda: Settings())

    tracker = st.SnapshotTracker()
    module = tmp_path / "mod.py"
    module.write_text("a = 1\n", encoding="utf-8")

    before = st.Snapshot(1.0, 0.0, 0.5, 0.0, 0.0)
    after = st.Snapshot(2.0, 1.0, 0.4, 0.0, 0.1, prompt="alpha")
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    tracker._context["after"] = {"prompt": "alpha", "files": [module]}

    delta = tracker.delta()
    assert delta["regression"] is False

    ckpt_base = tmp_path / "checkpoints"
    dirs = list(ckpt_base.iterdir())
    assert len(dirs) == 1
    ckpt_file = dirs[0] / module.name
    assert ckpt_file.exists()

    conf = json.loads((tmp_path / "strategy_confidence.json").read_text())
    assert conf["alpha"] == 1

    assert st.get_best_checkpoint(module) == ckpt_file
