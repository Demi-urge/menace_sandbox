import pytest

import pytest
import menace_sandbox.roi_tracker as rt
from menace_sandbox.roi_tracker import ROITracker


def test_calculate_raroi_basic():
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 20
    base_roi = 0.5
    base, raroi = tracker.calculate_raroi(base_roi, workflow_type="standard")
    assert base == base_roi
    assert raroi == pytest.approx(base_roi)


def test_calculate_raroi_with_risk():
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 20
    base_roi = 0.5
    tracker._last_errors_per_minute = 1.0
    failing = ["security", "alignment"]
    tracker._last_test_failures = failing
    base, raroi = tracker.calculate_raroi(base_roi, workflow_type="standard")
    assert base == base_roi
    error_prob = max(0.0, min(1.0, 1.0 / 10.0))
    rollback_probability = min(1.0, error_prob)
    expected = base_roi * (
        1.0 - rollback_probability * tracker.impact_severity("standard")
    )
    expected *= 0.5 if any(k in failing for k in rt.CRITICAL_SUITES) else 1.0
    assert raroi == pytest.approx(expected)


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
