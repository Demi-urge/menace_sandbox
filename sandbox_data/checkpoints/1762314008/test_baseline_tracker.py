import importlib.util
import sys
import types
from pathlib import Path

from dynamic_path_router import resolve_path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_DIR.parent))

root_pkg = types.ModuleType("menace")
root_pkg.__path__ = [str(MODULE_DIR.parent)]
sys.modules.setdefault("menace", root_pkg)

sub_pkg = types.ModuleType("menace.self_improvement")
sub_pkg.__path__ = [str(MODULE_DIR)]
sys.modules.setdefault("menace.self_improvement", sub_pkg)
sys.modules.setdefault("self_improvement", sub_pkg)

baseline_spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.baseline_tracker",
    resolve_path("self_improvement/baseline_tracker.py"),
)
baseline = importlib.util.module_from_spec(baseline_spec)
baseline_spec.loader.exec_module(baseline)

BaselineTracker = baseline.BaselineTracker


def _apply(tracker: BaselineTracker, score: float, margin: float = 0.0) -> tuple[bool, float]:
    """Helper returning acceptance decision and delta."""

    prev_avg = tracker.get("score")
    tracker.update(score=score)
    history = tracker.delta_history("score")
    delta = history[-1] if history else score - prev_avg
    return delta >= margin, delta


def test_accept_on_positive_delta():
    tracker = BaselineTracker(window=3)
    tracker.update(score=0.2)
    tracker.update(score=0.4)
    accepted, delta = _apply(tracker, 0.6)
    assert accepted and delta > 0


def test_reject_and_persist_on_negative_delta():
    tracker = BaselineTracker(window=3)
    tracker.update(score=0.8)
    tracker.update(score=0.7)
    accepted, delta = _apply(tracker, 0.6)
    assert not accepted and delta < 0
    assert tracker.current("score") == 0.6


def test_momentum_history_recorded():
    tracker = BaselineTracker(window=3)
    for roi in [1.0, 0.8, 1.2]:
        tracker.update(roi=roi)
    hist = tracker.to_dict()["momentum"]
    assert len(hist) == 3
    assert hist[-1] == tracker.momentum
    assert tracker.get("momentum") == sum(hist) / len(hist)


def test_roi_delta_and_success_history():
    tracker = BaselineTracker(window=3)
    tracker.update(roi=1.0)
    tracker.update(roi=2.0)
    tracker.update(roi=0.0)
    assert tracker.delta_history("roi") == [1.0, 1.0, -1.5]
    assert tracker.success_count == 2
    assert tracker.cycle_count == 3


def test_pass_rate_delta_from_moving_average():
    tracker = BaselineTracker(window=3)
    tracker.update(pass_rate=0.5)
    tracker.update(pass_rate=1.0)
    tracker.update(pass_rate=0.25)
    assert tracker.delta_history("pass_rate") == [0.5, 0.5, -0.5]


def test_current_and_delta_methods():
    tracker = BaselineTracker(window=3)
    tracker.update(roi=1.0)
    tracker.update(roi=3.0)
    # Latest value recorded
    assert tracker.current("roi") == 3.0
    # Moving average now 2.0, so deviation should be 1.0
    assert tracker.delta("roi") == 1.0

    tracker = BaselineTracker(window=3)
    tracker.update(pass_rate=0.5)
    tracker.update(pass_rate=1.0)
    assert tracker.current("pass_rate") == 1.0
    # Average is 0.75 -> delta should be 0.25
    assert tracker.delta("pass_rate") == 0.25
