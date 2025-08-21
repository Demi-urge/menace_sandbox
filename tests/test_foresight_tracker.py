import unittest.mock as mock
from types import SimpleNamespace
import pytest
from menace_sandbox.foresight_tracker import ForesightTracker

def test_record_cycle_metrics_window_and_order():
    tracker = ForesightTracker(max_cycles=3)
    for i in range(5):
        tracker.record_cycle_metrics("wf", {"roi_delta": float(i)})
    history = list(tracker.history["wf"])
    assert len(history) == 3
    assert [h["roi_delta"] for h in history] == [2.0, 3.0, 4.0]


def test_record_cycle_metrics_with_stubbed_roi_data():
    tracker = ForesightTracker(max_cycles=3)
    roi_tracker = SimpleNamespace(
        roi_history=[0.0],
        raroi_history=[0.0],
        workflow_confidence=lambda wf: 0.9,
    )

    roi_vals = [1, 3, 6, 10, 15]
    raroi_vals = [0.5, 1.5, 3.0, 5.0, 7.5]
    for rv, rr in zip(roi_vals, raroi_vals):
        roi_tracker.roi_history.append(rv)
        roi_tracker.raroi_history.append(rr)
        roi_delta = roi_tracker.roi_history[-1] - roi_tracker.roi_history[-2]
        raroi_delta = roi_tracker.raroi_history[-1] - roi_tracker.raroi_history[-2]
        tracker.record_cycle_metrics(
            "wf",
            {
                "roi_delta": roi_delta,
                "raroi_delta": raroi_delta,
                "confidence": roi_tracker.workflow_confidence("wf"),
                "resilience": 1.0,
                "scenario_degradation": 0.0,
            },
        )

    history = list(tracker.history["wf"])
    assert len(history) == 3
    assert [h["roi_delta"] for h in history] == [3.0, 4.0, 5.0]

def test_get_trend_curve_outputs():
    tracker = ForesightTracker()
    for val in [1, 2, 3, 4]:
        tracker.record_cycle_metrics("lin", {"roi_delta": val})
    slope, second_derivative, _ = tracker.get_trend_curve("lin")
    assert slope == pytest.approx(1.0)
    assert second_derivative == pytest.approx(0.0)

    tracker2 = ForesightTracker()
    for val in [1, 4, 9]:
        tracker2.record_cycle_metrics("quad", {"roi_delta": val})
    slope2, second_derivative2, _ = tracker2.get_trend_curve("quad")
    assert slope2 == pytest.approx(4.0)
    assert second_derivative2 == pytest.approx(2.0)


def test_get_trend_curve_unknown_workflow_id():
    tracker = ForesightTracker()
    assert tracker.get_trend_curve("missing") == (0.0, 0.0, 0.0)

def test_is_stable_scenarios():
    tracker = ForesightTracker()
    for val in [1, 1.5, 2, 2.5, 3]:
        tracker.record_cycle_metrics("wf", {"roi_delta": val})
    assert tracker.is_stable("wf")

    tracker2 = ForesightTracker()
    for val in [3, 2, 1]:
        tracker2.record_cycle_metrics("wf", {"roi_delta": val})
    assert not tracker2.is_stable("wf")


def test_is_stable_edge_cases():
    tracker = ForesightTracker()
    assert not tracker.is_stable("unknown")

    tracker_single = ForesightTracker()
    tracker_single.record_cycle_metrics("wf", {"roi_delta": 1.0})
    assert not tracker_single.is_stable("wf")

    tracker_vol = ForesightTracker()
    for val in [0, 3, 0, 3]:
        tracker_vol.record_cycle_metrics("wf", {"roi_delta": val})
    slope, _, volatility = tracker_vol.get_trend_curve("wf")
    assert slope > 0
    assert volatility > 1.0
    assert not tracker_vol.is_stable("wf")

def test_cycle_integration_calls_record_metrics(monkeypatch):
    tracker = ForesightTracker()
    record_mock = mock.MagicMock()
    monkeypatch.setattr(tracker, "record_cycle_metrics", record_mock)

    roi_tracker = SimpleNamespace(
        roi_history=[0.0, 1.0],
        raroi_history=[0.0, 0.5],
        workflow_confidence=lambda wf: 0.75,
    )

    ctx = SimpleNamespace(foresight_tracker=tracker, workflow_id="wf")
    resilience = 100.0
    metrics = {"scenario_degradation": 0.0}
    wf_id = getattr(ctx, "workflow_id", "_global")

    roi_delta = roi_tracker.roi_history[-1] - roi_tracker.roi_history[-2]
    raroi_delta = roi_tracker.raroi_history[-1] - roi_tracker.raroi_history[-2]
    conf_val = roi_tracker.workflow_confidence(wf_id)

    ctx.foresight_tracker.record_cycle_metrics(
        wf_id,
        {
            "roi_delta": roi_delta,
            "raroi_delta": raroi_delta,
            "confidence": conf_val,
            "resilience": resilience,
            "scenario_degradation": metrics["scenario_degradation"],
        },
    )

    record_mock.assert_called_once()
    args, _ = record_mock.call_args
    assert args[0] == "wf"
    assert args[1]["roi_delta"] == roi_delta
    assert args[1]["raroi_delta"] == raroi_delta
