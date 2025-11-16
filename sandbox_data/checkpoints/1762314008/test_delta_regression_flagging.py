import importlib
import types
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault("dynamic_path_router", types.SimpleNamespace(resolve_path=lambda p: Path(p)))
sys.modules.setdefault("audit_logger", types.SimpleNamespace(log_event=lambda *a, **k: None))
sys.modules.setdefault(
    "snapshot_history_db",
    types.SimpleNamespace(
        log_regression=lambda *a, **k: None,
        record_snapshot=lambda *a, **k: 1,
        record_delta=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "module_graph_analyzer",
    types.SimpleNamespace(
        build_import_graph=lambda root: types.SimpleNamespace(
            number_of_nodes=lambda: 0, number_of_edges=lambda: 0
        )
    ),
)

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

tracker_mod = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
from menace_sandbox.sandbox_settings import SandboxSettings


def test_delta_regression_flagging(monkeypatch, tmp_path):
    settings = SandboxSettings(
        sandbox_data_dir=str(tmp_path),
        snapshot_metrics=["roi", "entropy"],
        roi_drop_threshold=-0.05,
        entropy_regression_threshold=0.1,
    )
    monkeypatch.setattr(tracker_mod, "SandboxSettings", lambda: settings)
    tracker = tracker_mod.SnapshotTracker()

    calls: list[dict] = []
    monkeypatch.setattr(
        tracker_mod, "log_regression", lambda p, d, delta: calls.append(delta)
    )
    monkeypatch.setattr(tracker_mod, "record_delta", lambda *a, **k: None)
    monkeypatch.setattr(tracker_mod, "audit_log_event", lambda *a, **k: None)

    # ROI drop above threshold should not trigger regression
    before = tracker_mod.Snapshot(1.0, 0.0, 0.5, 0.0, 0.1, timestamp=0.0)
    after = tracker_mod.Snapshot(0.97, 0.0, 0.5, 0.0, 0.1, timestamp=1.0)
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    tracker._context["after"] = {}
    delta = tracker.delta()
    assert not delta["regression"]
    assert not calls

    # ROI drop below threshold should flag regression
    before = tracker_mod.Snapshot(1.0, 0.0, 0.5, 0.0, 0.1, timestamp=0.0)
    after = tracker_mod.Snapshot(0.9, 0.0, 0.5, 0.0, 0.1, timestamp=1.0)
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    delta = tracker.delta()
    assert delta["regression"]
    assert calls

    calls.clear()

    # Entropy increase below threshold should not flag regression
    before = tracker_mod.Snapshot(1.0, 0.0, 0.5, 0.0, 0.1, timestamp=0.0)
    after = tracker_mod.Snapshot(1.0, 0.0, 0.55, 0.0, 0.1, timestamp=1.0)
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    delta = tracker.delta()
    assert not delta["regression"]
    assert not calls

    # Entropy increase above threshold should flag regression
    before = tracker_mod.Snapshot(1.0, 0.0, 0.5, 0.0, 0.1, timestamp=0.0)
    after = tracker_mod.Snapshot(1.0, 0.0, 0.7, 0.0, 0.1, timestamp=1.0)
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    delta = tracker.delta()
    assert delta["regression"]
    assert calls
