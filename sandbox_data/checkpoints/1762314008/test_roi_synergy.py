import pytest
import importlib.util
import sys
import types
from pathlib import Path

if "menace.roi_tracker" in sys.modules:
    import menace.roi_tracker as rt
else:  # pragma: no cover - allow running file directly
    spec = importlib.util.spec_from_file_location(
        "menace.roi_tracker",
        Path(__file__).resolve().parents[1] / "roi_tracker.py",  # path-ignore
    )
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.roi_tracker = rt
    sys.modules["menace.roi_tracker"] = rt


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


def test_synergy_security_and_efficiency(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 6):
        roi = float(i)
        sec = float(2 * i)
        eff = float(3 * i)
        tracker.update(
            0.0,
            roi,
            metrics={
                "security_score": sec,
                "efficiency": eff,
                "synergy_security_score": 0.65 * i,
                "synergy_efficiency": 1.6 * i,
            },
        )

    monkeypatch.setattr(tracker, "forecast", lambda: (6.0, (0.0, 0.0)))

    def fake_forecast_metric(name):
        vals = {
            "security_score": (12.0, (0.0, 0.0)),
            "efficiency": (18.0, (0.0, 0.0)),
        }
        return vals.get(name, (0.0, (0.0, 0.0)))

    monkeypatch.setattr(tracker, "forecast_metric", fake_forecast_metric)

    sec_pred = tracker.predict_synergy_metric("security_score")
    eff_pred = tracker.predict_synergy_metric("efficiency")
    assert sec_pred == pytest.approx(0.65 * 6.0)
    assert eff_pred == pytest.approx(1.6 * 6.0)


def test_synergy_security_efficiency_insufficient():
    tracker = rt.ROITracker()
    assert tracker.predict_synergy_metric("security_score") == 0.0
    assert tracker.predict_synergy_metric("efficiency") == 0.0
    tracker.update(
        0.0,
        1.0,
        metrics={
            "security_score": 2.0,
            "efficiency": 3.0,
            "synergy_security_score": 0.65,
            "synergy_efficiency": 1.6,
        },
    )
    assert tracker.predict_synergy_metric("security_score") == 0.0
    assert tracker.predict_synergy_metric("efficiency") == 0.0
