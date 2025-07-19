import pytest
import menace.roi_tracker as rt


def test_predict_synergy_basic(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(5):
        roi = float(i)
        metric = float(i * 2)
        synergy = 0.5 * roi + 0.2 * metric
        tracker.update(0.0, roi, metrics={"m": metric, "synergy_roi": synergy})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (10.0, (0.0, 0.0)))

    pred = tracker.predict_synergy()
    assert pred == pytest.approx(0.5 * 5.0 + 0.2 * 10.0)


def test_predict_synergy_insufficient_data():
    tracker = rt.ROITracker()
    assert tracker.predict_synergy() == 0.0
    tracker.update(0.0, 1.0, metrics={"m": 1.0, "synergy_roi": 0.5})
    assert tracker.predict_synergy() == 0.0


def test_predict_synergy_metric_basic(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(5):
        roi = float(i)
        metric = float(i * 2)
        synergy = 0.1 * roi + 0.3 * metric
        tracker.update(0.0, roi, metrics={"m": metric, "synergy_m": synergy})

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (10.0, (0.0, 0.0)))

    pred = tracker.predict_synergy_metric("m")
    assert pred == pytest.approx(0.1 * 5.0 + 0.3 * 10.0)


def test_predict_synergy_metric_insufficient_data():
    tracker = rt.ROITracker()
    assert tracker.predict_synergy_metric("m") == 0.0
    tracker.update(0.0, 1.0, metrics={"m": 1.0, "synergy_m": 0.2})
    assert tracker.predict_synergy_metric("m") == 0.0


def test_predict_synergy_linear_simple(monkeypatch):
    tracker = rt.ROITracker()
    data = [0.1, 0.2, 0.1]
    for val in data:
        tracker.update(0.0, val, metrics={"synergy_roi": 0.5 * val})

    # Forecast ROI for the next step while leaving metrics unchanged
    monkeypatch.setattr(tracker, "forecast", lambda: (0.5, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (0.0, (0.0, 0.0)))

    pred = tracker.predict_synergy()
    assert pred == pytest.approx(0.25)


def test_predict_synergy_safety_rating(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(5):
        roi = float(i)
        rating = float(i * 2)
        synergy = 0.4 * roi + 0.2 * rating
        tracker.update(
            0.0,
            roi,
            metrics={"safety_rating": rating, "synergy_safety_rating": synergy},
        )

    monkeypatch.setattr(tracker, "forecast", lambda: (5.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (12.0, (0.0, 0.0)))

    pred = tracker.predict_synergy_safety_rating()
    assert pred == pytest.approx(0.4 * 5.0 + 0.2 * 12.0, rel=0.2)


def test_predict_synergy_metric_constant(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(4):
        tracker.update(0.0, float(i), metrics={"m": i * 2, "synergy_m": 0.1})

    monkeypatch.setattr(tracker, "forecast", lambda: (3.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (6.0, (0.0, 0.0)))

    assert tracker.predict_synergy_metric("m") == pytest.approx(0.1)


def test_predict_synergy_metric_negative(monkeypatch):
    tracker = rt.ROITracker()
    data = [-0.1, -0.15, -0.2]
    for i, val in enumerate(data, 1):
        tracker.update(0.0, 0.2 * i, metrics={"m": 0.5 * i, "synergy_m": val})

    monkeypatch.setattr(tracker, "forecast", lambda: (1.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (0.5, (0.0, 0.0)))

    assert tracker.predict_synergy_metric("m") < 0
