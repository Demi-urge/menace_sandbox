import numpy as np

from menace_sandbox.foresight_tracker import ForesightTracker


def test_records_truncated_to_window():
    tracker = ForesightTracker(window=3)
    for i in range(5):
        tracker.record_cycle_metrics("wf", {"m": float(i)})

    history = tracker.history["wf"]
    assert len(history) == 3
    assert [entry["m"] for entry in history] == [2.0, 3.0, 4.0]


def test_get_trend_curve_expected_values():
    tracker = ForesightTracker()

    # Linear series: slope of 1 and zero curvature
    for v in [0.0, 1.0, 2.0, 3.0]:
        tracker.record_cycle_metrics("linear", {"m": v})
    slope, second_derivative, _ = tracker.get_trend_curve("linear")
    assert np.isclose(slope, 1.0)
    assert np.isclose(second_derivative, 0.0)

    # Quadratic series: slope 3 and second derivative 2
    for v in [0.0, 1.0, 4.0, 9.0]:
        tracker.record_cycle_metrics("quadratic", {"m": v})
    slope_q, second_derivative_q, _ = tracker.get_trend_curve("quadratic")
    assert np.isclose(slope_q, 3.0)
    assert np.isclose(second_derivative_q, 2.0)


def test_is_stable_considers_trend_and_volatility():
    tracker = ForesightTracker(volatility_threshold=5.0)

    for v in [0.0, 1.0, 2.0, 3.0]:
        tracker.record_cycle_metrics("pos", {"m": v})
    assert tracker.is_stable("pos")

    for v in [3.0, 2.0, 1.0, 0.0]:
        tracker.record_cycle_metrics("neg", {"m": v})
    assert not tracker.is_stable("neg")

    for v in [0.0, 10.0, 0.0, 10.0]:
        tracker.record_cycle_metrics("vol", {"m": v})
    assert not tracker.is_stable("vol")


def test_to_from_dict_preserves_order_and_window():
    tracker = ForesightTracker(window=3)
    for v in [0.0, 1.0, 2.0, 3.0]:
        tracker.record_cycle_metrics("wf", {"m": v})

    data = tracker.to_dict()
    assert data["history"]["wf"] == [{"m": 1.0}, {"m": 2.0}, {"m": 3.0}]

    restored = ForesightTracker.from_dict(data, window=2, volatility_threshold=1.0)
    history = restored.history["wf"]
    assert [entry["m"] for entry in history] == [2.0, 3.0]
