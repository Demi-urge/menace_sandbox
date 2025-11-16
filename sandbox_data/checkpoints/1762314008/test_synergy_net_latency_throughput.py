import pytest
import roi_tracker as rt


def test_predict_synergy_network_latency_trend(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 6):
        roi = float(i)
        latency = 10.0 - i
        synergy = 0.2 * roi - 0.1 * latency
        tracker.update(0.0, roi, metrics={
            "network_latency": latency,
            "synergy_network_latency": synergy,
        })

    monkeypatch.setattr(tracker, "forecast", lambda: (6.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (5.0, (0.0, 0.0)))

    pred = tracker.predict_synergy_network_latency()
    assert pred == pytest.approx(0.2 * 6.0 - 0.1 * 5.0, rel=0.2)


def test_predict_synergy_network_latency_short_history():
    tracker = rt.ROITracker()
    tracker.update(0.0, 1.0, metrics={
        "network_latency": 9.0,
        "synergy_network_latency": 0.1,
    })
    assert tracker.predict_synergy_network_latency() == 0.0


def test_predict_synergy_network_latency_missing_metric(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 6):
        roi = float(i)
        synergy = 0.3 * roi
        tracker.update(0.0, roi, metrics={"synergy_network_latency": synergy})

    monkeypatch.setattr(tracker, "forecast", lambda: (6.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (0.0, (0.0, 0.0)))

    pred = tracker.predict_synergy_network_latency()
    assert pred == pytest.approx(0.3 * 6.0, rel=0.2)


def test_predict_synergy_throughput_trend(monkeypatch):
    tracker = rt.ROITracker()
    for i in range(1, 6):
        roi = float(i)
        throughput = float(i * 2)
        synergy = 0.1 * roi + 0.2 * throughput
        tracker.update(0.0, roi, metrics={
            "throughput": throughput,
            "synergy_throughput": synergy,
        })

    monkeypatch.setattr(tracker, "forecast", lambda: (6.0, (0.0, 0.0)))
    monkeypatch.setattr(tracker, "forecast_metric", lambda name: (12.0, (0.0, 0.0)))

    pred = tracker.predict_synergy_throughput()
    assert pred == pytest.approx(0.1 * 6.0 + 0.2 * 12.0, rel=0.2)
