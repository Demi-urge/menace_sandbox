import menace.roi_tracker as rt
import pytest


def test_synergy_revenue_storage():
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0, metrics={"revenue": 1.0, "synergy_revenue": 0.3})
    assert tracker.metrics_history["synergy_revenue"] == [0.3]


def test_synergy_revenue_forecast(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 5):
        val = float(i)
        tracker.update(0.0, val, metrics={"revenue": val, "synergy_revenue": val})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    assert tracker.predict_synergy_revenue() == pytest.approx(5.0)


def test_synergy_revenue_persistence(tmp_path):
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0, metrics={"revenue": 1.0, "synergy_revenue": 0.3})
    path = tmp_path / "hist.json"
    tracker.save_history(str(path))

    other = rt.ROITracker()
    other.load_history(str(path))
    assert other.synergy_metrics_history["synergy_revenue"] == [0.3]
