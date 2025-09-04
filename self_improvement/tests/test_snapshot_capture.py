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

baseline_mod = importlib.import_module("menace_sandbox.self_improvement.baseline_tracker")
tracker_mod = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
from menace_sandbox.sandbox_settings import SandboxSettings


def test_snapshot_capture(monkeypatch, tmp_path):
    settings = SandboxSettings(
        sandbox_repo_path=str(tmp_path),
        sandbox_data_dir=str(tmp_path / "data"),
        snapshot_metrics=["roi", "entropy"],
    )
    monkeypatch.setattr(tracker_mod, "SandboxSettings", lambda: settings)
    tracker = baseline_mod.BaselineTracker(window=3)
    monkeypatch.setattr(tracker_mod, "BASELINE_TRACKER", tracker)

    called = {"cg": False}

    monkeypatch.setattr(
        tracker_mod,
        "collect_snapshot_metrics",
        lambda files, settings=None: (0.5, 0.3),
    )

    def fake_call_graph(repo):  # pragma: no cover
        called["cg"] = True
        return 1.0

    monkeypatch.setattr(tracker_mod, "compute_call_graph_complexity", fake_call_graph)

    f = tmp_path / "m.py"
    f.write_text("print('hi')")

    snap = tracker_mod.capture("pre", [f], roi=1.0, sandbox_score=0.2)

    assert snap.entropy == 0.5
    assert tracker.current("roi") == 1.0
    assert tracker.current("entropy") == 0.5
    assert tracker.current("token_diversity") == 0.0
    assert called["cg"] is False
