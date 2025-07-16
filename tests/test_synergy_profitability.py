import menace.roi_tracker as rt
import pytest


def test_synergy_profitability_storage():
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0, metrics={"profitability": 1.0, "synergy_profitability": 0.2})
    assert tracker.metrics_history["synergy_profitability"] == [0.2]


def test_synergy_profitability_forecast(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 5):
        val = float(i)
        tracker.update(0.0, val, metrics={"profitability": val, "synergy_profitability": val})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    assert tracker.predict_synergy_profitability() == pytest.approx(5.0)


def test_synergy_projected_lucrativity_forecast(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 5):
        tracker.update(0.0, float(i), metrics={
            "projected_lucrativity": float(i),
            "synergy_projected_lucrativity": float(i),
        })

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    assert tracker.predict_synergy_projected_lucrativity() == pytest.approx(5.0)
