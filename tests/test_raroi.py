import pytest
import pytest
import menace_sandbox.roi_tracker as rt
from menace_sandbox.roi_tracker import ROITracker


def test_raroi_formula(monkeypatch):
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 10
    base_roi = 2.0
    monkeypatch.setattr(rt.np, "std", lambda arr: 0.0)
    monkeypatch.setattr(rt.ROITracker, "impact_severity", lambda self, wf: 0.4)
    failing = ["security"]
    tracker._last_errors_per_minute = 2.5
    tracker._last_test_failures = failing
    base, raroi = tracker.calculate_raroi(base_roi, workflow_type="standard")
    expected = base_roi * (1 - 0.25 * 0.4)
    expected *= 0.5 if any(k in failing for k in rt.CRITICAL_SUITES) else 1.0
    assert base == base_roi
    assert raroi == pytest.approx(expected)


def test_raroi_ranking_order(monkeypatch):
    tracker = ROITracker()
    monkeypatch.setattr(rt.np, "std", lambda arr: 0.0)
    tracker.update(
        0.0,
        2.0,
        modules=["a"],
        metrics={"errors_per_minute": 0.0, "security": True},
    )
    tracker.update(
        0.0,
        3.0,
        modules=["b"],
        metrics={"errors_per_minute": 8.0, "security": True},
    )
    ranking = tracker.rankings()
    assert ranking[0][0] == "a"
    assert ranking[0][2] < ranking[1][2]
    assert ranking[0][1] > ranking[1][1]
