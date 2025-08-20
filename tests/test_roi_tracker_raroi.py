import pytest

import numpy as np
import menace.roi_tracker as rt
from menace.roi_tracker import ROITracker
import menace.self_test_service as sts


@pytest.fixture
def failing_critical_tests():
    """Set names of failing critical test suites and reset afterwards."""
    def _set(names: list[str]) -> None:
        sts.set_failed_critical_tests(names)
    yield _set
    sts.set_failed_critical_tests([])


def test_raroi_with_known_factors(failing_critical_tests):
    tracker = ROITracker()
    tracker.roi_history = [1.0, 1.5, 2.0]
    failing_critical_tests(["security"])
    base_roi = 1.0
    base, raroi, _ = tracker.calculate_raroi(
        base_roi,
        rollback_prob=0.2,
        impact_severity=0.5,
        workflow_type="standard",
    )
    instability = np.std(tracker.roi_history[-tracker.window :])
    expected = (
        base_roi
        * (1.0 - 0.2 * 0.5)
        * max(0.0, 1.0 - instability)
        * rt.CRITICAL_TEST_PENALTIES["security"]
    )
    assert base == base_roi
    assert raroi == pytest.approx(expected)


def test_raroi_empty_history_and_metrics(failing_critical_tests):
    tracker = ROITracker()
    failing_critical_tests([])
    base, raroi, _ = tracker.calculate_raroi(2.0, workflow_type="standard")
    assert base == 2.0
    assert raroi == pytest.approx(2.0)


def test_raroi_extreme_values(failing_critical_tests):
    tracker = ROITracker()
    tracker.roi_history = [10.0, -10.0, 10.0]
    failing_critical_tests(["security", "alignment"])
    base, raroi, _ = tracker.calculate_raroi(
        1.0,
        rollback_prob=1.0,
        impact_severity=1.0,
        workflow_type="standard",
    )
    assert base == 1.0
    assert raroi == 0.0


def test_raroi_with_error_metrics(failing_critical_tests):
    tracker = ROITracker()
    failing_critical_tests([])
    base_roi = 2.0
    metrics = {"errors_per_minute": 5.0, "error_threshold": 10.0}
    base, raroi, _ = tracker.calculate_raroi(
        base_roi, workflow_type="standard", metrics=metrics
    )
    expected = base_roi * (1.0 - (5.0 / 10.0) * 0.5)
    assert base == base_roi
    assert raroi == pytest.approx(expected)
    assert raroi < base_roi


def test_raroi_multiple_failing_suites_param(failing_critical_tests):
    tracker = ROITracker()
    tracker.roi_history = [1.0, 1.5, 2.0]
    failing_critical_tests([])
    base_roi = 1.0
    failing = ["security", "alignment"]
    base, raroi, _ = tracker.calculate_raroi(
        base_roi,
        rollback_prob=0.2,
        impact_severity=0.5,
        workflow_type="standard",
        failing_tests=failing,
    )
    instability = np.std(tracker.roi_history[-tracker.window :])
    penalty = (
        rt.CRITICAL_TEST_PENALTIES["security"]
        * rt.CRITICAL_TEST_PENALTIES["alignment"]
    )
    expected = (
        base_roi
        * (1.0 - 0.2 * 0.5)
        * max(0.0, 1.0 - instability)
        * penalty
    )
    assert base == base_roi
    assert raroi == pytest.approx(expected)
