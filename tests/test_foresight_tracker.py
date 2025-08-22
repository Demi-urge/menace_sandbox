
import numpy as np
import pytest

from foresight_tracker import ForesightTracker


def test_records_truncated_to_max_cycles():
    tracker = ForesightTracker(max_cycles=3)
    for i in range(5):
        tracker.record_cycle_metrics("wf", {"m": float(i)})

    history = tracker.history["wf"]
    assert len(history) == 3
    assert [entry["m"] for entry in history] == [2.0, 3.0, 4.0]


def test_get_trend_curve_expected_values():
    tracker = ForesightTracker()

    # Linear series: slope of 1 and zero curvature
    for v in [0.0, 1.0, 2.0, 3.0]:
        tracker.record_cycle_metrics("linear", {"m": v})
    slope, second_derivative, _ = tracker.get_trend_curve("linear")
    assert np.isclose(slope, 1.0)
    assert np.isclose(second_derivative, 0.0)

    # Quadratic series: slope 3 and second derivative 2
    for v in [0.0, 1.0, 4.0, 9.0]:
        tracker.record_cycle_metrics("quadratic", {"m": v})
    slope_q, second_derivative_q, _ = tracker.get_trend_curve("quadratic")
    assert np.isclose(slope_q, 3.0)
    assert np.isclose(second_derivative_q, 2.0)


def test_is_stable_considers_trend_and_volatility():
    tracker = ForesightTracker(volatility_threshold=5.0)

    for v in [0.0, 1.0, 2.0, 3.0]:
        tracker.record_cycle_metrics("pos", {"m": v})
    assert tracker.is_stable("pos")

    for v in [3.0, 2.0, 1.0, 0.0]:
        tracker.record_cycle_metrics("neg", {"m": v})
    assert not tracker.is_stable("neg")

    for v in [0.0, 10.0, 0.0, 10.0]:
        tracker.record_cycle_metrics("vol", {"m": v})
    assert not tracker.is_stable("vol")


def test_to_dict_from_dict_roundtrip_configuration_and_history():
    tracker = ForesightTracker(max_cycles=4, volatility_threshold=2.5)
    for v in [0.0, 1.0, 2.0, 3.0]:
        tracker.record_cycle_metrics("wf", {"m": v})

    data = tracker.to_dict()
    assert data["window"] == 4
    assert data["volatility_threshold"] == 2.5
    assert data["history"]["wf"] == [
        {"m": 0.0},
        {"m": 1.0},
        {"m": 2.0},
        {"m": 3.0},
    ]

    restored = ForesightTracker.from_dict(data)
    assert restored.max_cycles == 4
    assert np.isclose(restored.volatility_threshold, 2.5)
    history = restored.history["wf"]
    assert [entry["m"] for entry in history] == [0.0, 1.0, 2.0, 3.0]


def test_from_dict_allows_overrides_and_truncates_history():
    tracker = ForesightTracker(max_cycles=5, volatility_threshold=3.0)
    for i in range(6):
        tracker.record_cycle_metrics("wf", {"m": float(i)})

    data = tracker.to_dict()
    restored = ForesightTracker.from_dict(
        data, N=3, volatility_threshold=1.0
    )
    assert restored.max_cycles == 3
    assert np.isclose(restored.volatility_threshold, 1.0)
    history = restored.history["wf"]
    assert [entry["m"] for entry in history] == [3.0, 4.0, 5.0]
    assert len(history) == 3


def test_init_accepts_aliases():
    tracker_n = ForesightTracker(N=5)
    tracker_w = ForesightTracker(window=4)
    assert tracker_n.max_cycles == 5
    assert tracker_w.max_cycles == 4


def test_to_dict_serializes_window():
    tracker = ForesightTracker(max_cycles=7)
    data = tracker.to_dict()
    assert data["window"] == 7
    assert "max_cycles" not in data


def test_from_dict_reads_legacy_keys():
    data_n = {"N": 2, "history": {"wf": [{"m": 1.0}, {"m": 2.0}]}}
    tracker_n = ForesightTracker.from_dict(data_n)
    assert tracker_n.max_cycles == 2
    assert [e["m"] for e in tracker_n.history["wf"]] == [1.0, 2.0]

    data_w = {"window": 3, "history": {"wf": [{"m": 1.0}, {"m": 2.0}, {"m": 3.0}]}}
    tracker_w = ForesightTracker.from_dict(data_w)
    assert tracker_w.max_cycles == 3
    assert [e["m"] for e in tracker_w.history["wf"]] == [1.0, 2.0, 3.0]


def test_is_cold_start_requires_three_cycles():
    tracker = ForesightTracker()
    assert tracker.is_cold_start("wf")
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.1})
    assert tracker.is_cold_start("wf")
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.2})
    assert tracker.is_cold_start("wf")
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.3})
    assert not tracker.is_cold_start("wf")


def test_is_cold_start_when_roi_delta_missing():
    tracker = ForesightTracker()
    for _ in range(3):
        tracker.record_cycle_metrics("wf", {"raroi_delta": 0.1})
    assert tracker.is_cold_start("wf")
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.2})
    assert not tracker.is_cold_start("wf")


def test_is_cold_start_when_roi_delta_zero():
    tracker = ForesightTracker()
    for _ in range(3):
        tracker.record_cycle_metrics("wf", {"roi_delta": 0.0})
    assert tracker.is_cold_start("wf")
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.5})
    assert not tracker.is_cold_start("wf")


def test_capture_from_roi_records_latest_metrics():
    ft = ForesightTracker()

    class DummyROITracker:
        def __init__(self):
            self.roi_history = [0.1]
            self.raroi_history = [0.2, 0.5]
            self.confidence_history = [0.9]
            self.metrics_history = {"synergy_resilience": [0.8]}

        def scenario_degradation(self):
            return 0.3

    dummy = DummyROITracker()
    ft.capture_from_roi(dummy, "wf")

    entry = ft.history["wf"][0]
    assert entry["roi_delta"] == 0.1
    assert entry["raroi_delta"] == 0.3  # 0.5 - 0.2
    assert entry["confidence"] == 0.9
    assert entry["resilience"] == 0.8
    assert entry["scenario_degradation"] == 0.3


def test_capture_from_roi_blends_template_and_real_roi(tracker_with_templates):
    class DummyROITracker:
        def __init__(self):
            self.roi_history = []
            self.raroi_history = []
            self.confidence_history = []
            self.metrics_history = {"synergy_resilience": []}

        def scenario_degradation(self):
            return 0.0

    dummy = DummyROITracker()

    # No prior cycles -> pure template value
    tracker_with_templates.capture_from_roi(dummy, "wf")
    # One cycle -> blend of real and template ROI
    dummy.roi_history.append(1.0)
    tracker_with_templates.capture_from_roi(dummy, "wf")
    # Additional cycles to reach warm state where alpha == 1.0
    for _ in range(4):
        dummy.roi_history.append(1.0)
        tracker_with_templates.capture_from_roi(dummy, "wf")

    history = tracker_with_templates.history["wf"]
    assert history[0]["roi_delta"] == pytest.approx(0.5)
    assert history[1]["roi_delta"] == pytest.approx(0.6)
    assert history[-1]["roi_delta"] == pytest.approx(1.0)


def test_to_dict_from_dict_roundtrip_after_capture_from_roi():
    ft = ForesightTracker()

    class DummyROITracker:
        def __init__(self):
            self.roi_history: list[float] = []
            self.raroi_history: list[float] = []
            self.confidence_history: list[float] = []
            self.metrics_history = {"synergy_resilience": []}

        def scenario_degradation(self) -> float:
            return 0.0

    dummy = DummyROITracker()
    for v in [0.0, 0.1, 0.2]:
        dummy.roi_history.append(v)
        dummy.raroi_history.append(v)
        dummy.confidence_history.append(0.9)
        dummy.metrics_history["synergy_resilience"].append(0.8)
        ft.capture_from_roi(dummy, "wf")

    data = ft.to_dict()
    restored = ForesightTracker.from_dict(data)
    assert restored.to_dict() == data


def test_cold_start_persists_through_serialization(
    tracker_with_templates, foresight_templates
):
    class DummyROITracker:
        def __init__(self):
            self.roi_history: list[float] = []
            self.raroi_history: list[float] = []
            self.confidence_history: list[float] = []
            self.metrics_history = {"synergy_resilience": []}

        def scenario_degradation(self) -> float:
            return 0.0

    dummy = DummyROITracker()
    for _ in range(2):
        dummy.roi_history.append(1.0)
        tracker_with_templates.capture_from_roi(dummy, "wf")

    assert tracker_with_templates.is_cold_start("wf")

    data = tracker_with_templates.to_dict()
    restored = ForesightTracker.from_dict(data)
    restored.template_config_path = foresight_templates
    restored.templates = None

    assert restored.is_cold_start("wf")

    dummy.roi_history.append(1.0)
    restored.capture_from_roi(dummy, "wf")
    assert restored.history["wf"][-1]["roi_delta"] == pytest.approx(0.7)


def test_capture_from_roi_self_improvement_cycle_affects_stability():
    ft = ForesightTracker(volatility_threshold=0.5)

    class DummyROITracker:
        def __init__(self):
            self.roi_history: list[float] = []
            self.raroi_history: list[float] = []
            self.confidence_history: list[float] = []
            self.metrics_history = {"synergy_resilience": []}

        def scenario_degradation(self) -> float:
            return 0.0

    # Stable improvement should be considered stable
    stable = DummyROITracker()
    for v in [0.0, 0.1, 0.2, 0.3]:
        stable.roi_history.append(v)
        stable.raroi_history.append(v)
        stable.confidence_history.append(0.9)
        stable.metrics_history["synergy_resilience"].append(0.8)
        ft.capture_from_roi(stable, "stable")

    history = ft.history["stable"]
    assert [e["roi_delta"] for e in history] == [0.0, 0.1, 0.2, 0.3]
    assert ft.is_stable("stable")

    # Negative trend should be unstable
    negative = DummyROITracker()
    for v in [0.3, 0.2, 0.1, 0.0]:
        negative.roi_history.append(v)
        negative.raroi_history.append(v)
        negative.confidence_history.append(0.9)
        negative.metrics_history["synergy_resilience"].append(0.8)
        ft.capture_from_roi(negative, "negative")
    assert not ft.is_stable("negative")

    # High volatility should be unstable
    volatile = DummyROITracker()
    for v in [0.0, 1.0, 0.0, 1.0]:
        volatile.roi_history.append(v)
        volatile.raroi_history.append(v)
        volatile.confidence_history.append(0.9)
        volatile.metrics_history["synergy_resilience"].append(0.8)
        ft.capture_from_roi(volatile, "volatile")
    assert not ft.is_stable("volatile")
