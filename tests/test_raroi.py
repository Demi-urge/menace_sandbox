import pytest
import menace_sandbox.roi_tracker as rt
from menace_sandbox.roi_tracker import ROITracker


def test_raroi_formula(monkeypatch):
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 10
    base_roi = 2.0
    monkeypatch.setattr(tracker, "_rollback_probability", lambda metrics: 0.25)
    monkeypatch.setattr(tracker, "_impact_severity", lambda workflow_type: 0.4)
    monkeypatch.setattr(tracker, "_safety_factor", lambda metrics: 0.8)
    base, raroi = tracker.calculate_raroi(base_roi, "standard", {})
    expected = base_roi * (1 - 0.25 * 0.4) * 1.0 * 0.8
    assert base == base_roi
    assert raroi == pytest.approx(expected)


def test_raroi_ranking_order(monkeypatch):
    tracker = ROITracker()
    monkeypatch.setattr(rt.np, "std", lambda arr: 0.0)
    monkeypatch.setattr(tracker, "_rollback_probability", lambda metrics: metrics.get("risk", 0.0))
    monkeypatch.setattr(tracker, "_impact_severity", lambda workflow_type: 1.0)
    monkeypatch.setattr(tracker, "_safety_factor", lambda metrics: metrics.get("safety", 1.0))
    tracker.update(0.0, 2.0, modules=["a"], metrics={"risk": 0.0, "safety": 1.0})
    tracker.update(0.0, 3.0, modules=["b"], metrics={"risk": 0.8, "safety": 1.0})
    ranking = tracker.rankings()
    assert ranking[0][0] == "a"
    assert ranking[0][2] < ranking[1][2]
    assert ranking[0][1] > ranking[1][1]
