import numpy as np
from menace_sandbox.foresight_tracker import ForesightTracker


def _metrics(val):
    return {
        "roi_delta": val,
        "raroi_delta": val,
        "confidence": val,
        "resilience": val,
        "scenario_degradation": val,
    }


def test_history_limit_and_serialization():
    tracker = ForesightTracker(max_cycles=3)
    for i in range(5):
        tracker.record_cycle_metrics("wf", _metrics(i))
    history = tracker.get_history("wf")
    assert len(history) == 3
    assert history[0]["roi_delta"] == 2

    data = tracker.to_dict()
    restored = ForesightTracker.from_dict(data, max_cycles=3)
    assert restored.get_history("wf") == history


def test_trend_curve_and_stability():
    tracker = ForesightTracker(volatility_threshold=2.0)
    values = [0, 1, 2, 3]
    for v in values:
        tracker.record_cycle_metrics("wf", _metrics(v))
    slope, second_derivative, volatility = tracker.get_trend_curve("wf")
    assert slope > 0
    assert np.isclose(second_derivative, 0.0)
    assert np.isclose(volatility, np.std(values, ddof=1))
    assert tracker.is_stable("wf")

    # Introduce volatility
    tracker.record_cycle_metrics("wf", _metrics(10))
    assert not tracker.is_stable("wf")
    assert not tracker.is_stable("wf", threshold=0.2)

