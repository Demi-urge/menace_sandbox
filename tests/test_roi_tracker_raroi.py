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
