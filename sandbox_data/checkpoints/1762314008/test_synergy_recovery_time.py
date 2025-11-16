import pytest
import menace.roi_tracker as rt


def test_synergy_recovery_time_storage():
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0, metrics={"recovery_time": 2.0, "synergy_recovery_time": -0.5})
    assert tracker.metrics_history["synergy_recovery_time"] == [-0.5]


def test_synergy_recovery_time_forecast(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 5):
        val = float(i)
        tracker.update(0.0, val, metrics={"recovery_time": val, "synergy_recovery_time": val})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    assert tracker.predict_synergy_recovery_time() == pytest.approx(5.0)
