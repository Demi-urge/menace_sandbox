import pytest
from menace_sandbox.roi_tracker import ROITracker


def test_calculate_raroi_basic():
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 20
    base_roi = 0.5
    metrics = {"errors_per_minute": 0.0, "safety_rating": 1.0, "security_score": 1.0}
    base, raroi = tracker.calculate_raroi(base_roi, "standard", metrics)
    assert base == base_roi
    assert raroi == pytest.approx(base_roi)


def test_calculate_raroi_with_risk():
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 20
    base_roi = 0.5
    metrics = {
        "errors_per_minute": 1.0,
        "safety_rating": 0.2,
        "security_score": 0.2,
    }
    base, raroi = tracker.calculate_raroi(base_roi, "standard", metrics)
    assert base == base_roi
    assert raroi < base_roi * 0.3


def test_update_records_raroi_and_rankings():
    tracker = ROITracker()
    metrics = {"errors_per_minute": 0.0, "safety_rating": 1.0, "security_score": 1.0}
    tracker.update(0.0, 1.0, modules=["m1"], metrics=metrics, category="standard")
    tracker.update(0.0, 0.5, modules=["m2"], metrics=metrics, category="standard")
    assert len(tracker.raroi_history) == 2
    assert tracker.raroi_history[0] == pytest.approx(tracker.roi_history[0])
    assert tracker.raroi_history[1] <= tracker.roi_history[1]
    ranking = tracker.rankings()
    assert ranking[0][0] == "m1"
    assert ranking[0][1] >= ranking[1][1]
    assert ranking[0][2] >= ranking[1][2]
