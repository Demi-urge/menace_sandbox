import importlib.util
import sys
import types

import pytest

from dynamic_path_router import resolve_path

# Prepare fake package hierarchy to satisfy relative imports
ROOT_DIR = resolve_path(".")
PACKAGE_DIR = resolve_path("self_improvement")
root_name = "menace_sandbox"
if root_name not in sys.modules:
    root_pkg = types.ModuleType(root_name)
    root_pkg.__path__ = [str(ROOT_DIR)]  # type: ignore[attr-defined]
    sys.modules[root_name] = root_pkg

package_name = f"{root_name}.self_improvement"
if package_name not in sys.modules:
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
    sys.modules[package_name] = pkg

# Load BaselineTracker without executing package __init__
spec = importlib.util.spec_from_file_location(
    f"{package_name}.baseline_tracker",
    resolve_path("self_improvement/baseline_tracker.py"),
)
baseline_tracker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline_tracker
spec.loader.exec_module(baseline_tracker)  # type: ignore[assignment]
BaselineTracker = baseline_tracker.BaselineTracker

# Load system_snapshot module
spec = importlib.util.spec_from_file_location(
    f"{package_name}.system_snapshot",
    resolve_path("self_improvement/system_snapshot.py"),
)
system_snapshot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = system_snapshot
spec.loader.exec_module(system_snapshot)  # type: ignore[assignment]


class DummyEngine:
    def __init__(self, tracker):
        self.roi_tracker = tracker
        self.last_prompt = "test prompt"
        self.module_paths = ["a.py", "b.py"]  # path-ignore


def test_capture_snapshot(monkeypatch, tmp_path):
    tracker = BaselineTracker()
    tracker.update(roi=1.5, entropy=0.25)
    engine = DummyEngine(tracker)

    # stub sandbox_results_logger.load_summary
    fake_logger = types.SimpleNamespace(load_summary=lambda: {"sandbox_score": 0.8})
    monkeypatch.setitem(sys.modules, "sandbox_results_logger", fake_logger)
    monkeypatch.setattr(system_snapshot, "sandbox_results_logger", fake_logger)

    # stub SandboxSettings to point at temporary repo
    settings = system_snapshot.SandboxSettings(sandbox_repo_path=str(tmp_path))
    monkeypatch.setattr(system_snapshot, "SandboxSettings", lambda: settings)

    # stub metrics._collect_metrics to control token_diversity
    def fake_collect(files, repo, settings=None):
        return ({}, 0, 0.0, 0, 0.0, 0.6)

    monkeypatch.setattr(system_snapshot._si_metrics, "_collect_metrics", fake_collect)
    monkeypatch.setattr(system_snapshot, "compute_call_graph_complexity", lambda repo: 2.0)

    snap = system_snapshot.capture_snapshot(engine)
    assert snap.roi == 1.5
    assert snap.sandbox_score == 0.8
    assert snap.entropy == 0.25
    assert snap.call_graph_complexity == 2.0
    assert snap.token_diversity == 0.6
    assert snap.metadata["prompt"] == "test prompt"
    assert snap.metadata["module_paths"] == ["a.py", "b.py"]  # path-ignore


def test_compare_snapshots():
    before = system_snapshot.SystemSnapshot(
        roi=1.0,
        sandbox_score=0.5,
        entropy=0.1,
        call_graph_complexity=1.0,
        token_diversity=0.2,
        timestamp=0.0,
        metadata={},
    )
    after = system_snapshot.SystemSnapshot(
        roi=2.0,
        sandbox_score=0.7,
        entropy=0.3,
        call_graph_complexity=1.5,
        token_diversity=0.25,
        timestamp=1.0,
        metadata={},
    )
    delta = system_snapshot.compare_snapshots(before, after)
    assert delta == {
        "roi": pytest.approx(1.0),
        "sandbox_score": pytest.approx(0.2),
        "entropy": pytest.approx(0.2),
        "call_graph_complexity": pytest.approx(0.5),
        "token_diversity": pytest.approx(0.05),
    }
