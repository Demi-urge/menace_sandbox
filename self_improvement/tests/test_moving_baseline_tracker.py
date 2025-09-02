import importlib

utils = importlib.import_module("menace.self_improvement.utils")
baseline = importlib.import_module("menace.self_improvement.baseline_tracker")
MovingBaselineTracker = utils.MovingBaselineTracker
BaselineTracker = baseline.BaselineTracker


def _apply(tracker: MovingBaselineTracker, score: float, margin: float = 0.0) -> tuple[bool, float]:
    """Helper returning acceptance decision and delta."""

    avg, _ = tracker.stats()
    delta = score - avg
    tracker.update(score)
    return delta >= margin, delta


def test_accept_on_positive_delta():
    tracker = MovingBaselineTracker(3)
    tracker.update(0.2)
    tracker.update(0.4)
    accepted, delta = _apply(tracker, 0.6)
    assert accepted and delta > 0


def test_reject_and_persist_on_negative_delta():
    tracker = MovingBaselineTracker(3)
    tracker.update(0.8)
    tracker.update(0.7)
    accepted, delta = _apply(tracker, 0.6)
    assert not accepted and delta < 0
    assert tracker.composite_history[-1] == 0.6


def test_momentum_history_recorded():
    tracker = BaselineTracker(window=3)
    for roi in [1.0, 0.8, 1.2]:
        tracker.update(roi=roi)
    hist = tracker.to_dict()["momentum"]
    assert len(hist) == 3
    assert hist[-1] == tracker.momentum
    assert tracker.get("momentum") == sum(hist) / len(hist)
