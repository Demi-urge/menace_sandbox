import numpy as np

from menace_sandbox.foresight_tracker import ForesightTracker


def test_history_limit():
    tracker = ForesightTracker(window=3, volatility_threshold=1.0)
    for i in range(5):
        tracker.record_cycle_metrics("wf", {"m": float(i)})

    history = tracker.history["wf"]
    assert len(history) == 3
    assert list(history)[0]["m"] == 2.0


def test_trend_curve_and_stability():
    tracker = ForesightTracker(volatility_threshold=2.0)
    values = [0.0, 1.0, 2.0, 3.0]
    for v in values:
        tracker.record_cycle_metrics("wf", {"m": v})

    slope, second_derivative, stability = tracker.get_trend_curve("wf")
    assert slope > 0
    assert np.isclose(second_derivative, 0.0)
    expected_std = np.std(values, ddof=1)
    expected_stability = 1 / (1 + expected_std)
    assert np.isclose(stability, expected_stability)
    assert tracker.is_stable("wf")

    tracker.record_cycle_metrics("wf", {"m": 10.0})
    assert not tracker.is_stable("wf")

