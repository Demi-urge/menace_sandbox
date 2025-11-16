import menace.roi_tracker as rt
import pytest


def test_synergy_risk_index_storage():
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0, metrics={"risk_index": 1.0, "synergy_risk_index": 0.5})
    assert tracker.metrics_history["synergy_risk_index"] == [0.5]


def test_synergy_risk_index_forecast(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 5):
        val = float(i)
        tracker.update(0.0, val, metrics={"risk_index": val, "synergy_risk_index": val})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    assert tracker.predict_synergy_risk_index() == pytest.approx(5.0)
