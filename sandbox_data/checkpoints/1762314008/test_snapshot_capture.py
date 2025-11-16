import importlib
import types
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

dyn_stub = types.ModuleType("dynamic_path_router")
dyn_stub.resolve_path = lambda p: Path(p)
dyn_stub.repo_root = lambda: Path(".")
dyn_stub.resolve_dir = lambda p: Path(p)
sys.modules["dynamic_path_router"] = dyn_stub
from dynamic_path_router import resolve_path
sandbox_pkg = types.ModuleType("sandbox_runner")
sandbox_pkg.bootstrap = types.SimpleNamespace(initialize_autonomous_sandbox=lambda: None)
sys.modules["sandbox_runner"] = sandbox_pkg
sys.modules["sandbox_runner.bootstrap"] = sandbox_pkg.bootstrap
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

    f = tmp_path / resolve_path("m.py")
    f.write_text("print('hi')")

    snap = tracker_mod.capture("pre", [f], roi=1.0, sandbox_score=0.2)

    assert snap.entropy == 0.5
    assert tracker.current("roi") == 1.0
    assert tracker.current("entropy") == 0.5
    assert tracker.current("token_diversity") == 0.0
    assert called["cg"] is False


def test_capture_with_repo_path(monkeypatch, tmp_path):
    settings = SandboxSettings(
        sandbox_repo_path=str(tmp_path),
        sandbox_data_dir=str(tmp_path / "data"),
        snapshot_metrics=["entropy", "token_diversity"],
    )
    monkeypatch.setattr(tracker_mod, "SandboxSettings", lambda: settings)
    tracker = baseline_mod.BaselineTracker(window=3)
    monkeypatch.setattr(tracker_mod, "BASELINE_TRACKER", tracker)

    def fake_collect(files, settings=None):
        count = sum(1 for _ in files)
        return float(count), float(count)

    monkeypatch.setattr(tracker_mod, "collect_snapshot_metrics", fake_collect)

    tracker_instance = tracker_mod.SnapshotTracker()

    (tmp_path / resolve_path("a.py")).write_text("a=1")
    (tmp_path / resolve_path("b.py")).write_text("b=2")

    snap1 = tracker_instance.capture(
        "before", {"roi": 0.0, "sandbox_score": 0.0}, repo_path=tmp_path
    )
    assert snap1.entropy == 2.0
    assert snap1.token_diversity == 2.0

    (tmp_path / resolve_path("c.py")).write_text("c=3")

    snap2 = tracker_instance.capture(
        "after", {"roi": 0.0, "sandbox_score": 0.0}, repo_path=tmp_path
    )
    assert snap2.entropy == 3.0
    assert snap2.token_diversity == 3.0

    delta = tracker_instance.delta()
    assert delta["entropy"] == 1.0
    assert delta["token_diversity"] == 1.0
