
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


def test_record_cycle_metrics_computes_stability_when_requested():
    tracker = ForesightTracker()
    tracker.record_cycle_metrics("wf", {"m": 1.0}, compute_stability=True)
    first = tracker.history["wf"][0]
    assert first["stability"] == 0.0

    tracker.record_cycle_metrics("wf", {"m": 2.0}, compute_stability=True)
    stabilities = [e["stability"] for e in tracker.history["wf"]]
    expected_std = np.std([0.5, 2.0], ddof=1)
    expected = 1.0 / (1.0 + expected_std)
    assert stabilities[-1] == pytest.approx(expected)


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


def test_is_cold_start_for_insufficient_cycles_or_zero_roi():
    tracker = ForesightTracker()
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.1})
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.2})
    assert tracker.is_cold_start("wf")  # fewer than three cycles

    tracker_zero = ForesightTracker()
    for _ in range(3):
        tracker_zero.record_cycle_metrics("wf", {"roi_delta": 0.0})
    assert tracker_zero.is_cold_start("wf")  # zero ROI across cycles


@pytest.mark.parametrize(
    "roi_values,threshold,expected",
    [
        ([0.1, 0.2, 0.3], 1.0, "Stable"),
        ([0.5, 0.4, 0.3], 1.0, "Slow decay"),
        ([0.0, 2.0, 0.0, 2.0], 1.0, "Volatile"),
        ([0.5, 0.0, -0.5], 1.0, "Immediate collapse risk"),
    ],
)
def test_predict_roi_collapse_risk_categories(roi_values, threshold, expected):
    tracker = ForesightTracker(volatility_threshold=threshold)
    for v in roi_values:
        tracker.record_cycle_metrics("wf", {"roi_delta": v})
    result = tracker.predict_roi_collapse("wf")
    assert result["risk"] == expected
    assert result["brittle"] is False


def test_predict_roi_collapse_brittle_flag():
    tracker = ForesightTracker()
    tracker.record_cycle_metrics("wf", {"roi_delta": 1.0}, scenario_degradation=0.0)
    tracker.record_cycle_metrics("wf", {"roi_delta": 1.0}, scenario_degradation=0.0)
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.0}, scenario_degradation=0.01)
    result = tracker.predict_roi_collapse("wf")
    assert result["brittle"] is True
    assert result["risk"] == "Immediate collapse risk"


def test_template_blending_and_cold_start(tmp_path):
    cfg = tmp_path / "foresight_templates.yaml"
    cfg.write_text(
        "profiles:\n  wf: basic\ntrajectories:\n  basic: [0.5, 0.5, 0.5, 0.5, 0.5]\n",
        encoding="utf8",
    )

    tracker = ForesightTracker(templates_path=cfg)

    class DummyROITracker:
        def __init__(self):
            self.roi_history = []
            self.raroi_history = []
            self.confidence_history = []
            self.metrics_history = {"synergy_resilience": []}

        def scenario_degradation(self):
            return 0.0

    dummy = DummyROITracker()

    dummy.roi_history.append(1.0)
    tracker.capture_from_roi(dummy, "wf")
    assert tracker.is_cold_start("wf")
    assert tracker.history["wf"][0]["roi_delta"] == pytest.approx(0.5)

    dummy.roi_history.append(1.0)
    tracker.capture_from_roi(dummy, "wf")
    assert tracker.is_cold_start("wf")
    assert tracker.history["wf"][1]["roi_delta"] == pytest.approx(0.6)

    dummy.roi_history.append(1.0)
    tracker.capture_from_roi(dummy, "wf")
    assert not tracker.is_cold_start("wf")
    assert tracker.history["wf"][2]["roi_delta"] == pytest.approx(0.7)


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
    assert entry["raw_roi_delta"] == 0.1


def test_capture_from_roi_records_entropy():
    ft = ForesightTracker()

    class DummyROITracker:
        def __init__(self):
            self.roi_history = [0.1]
            self.raroi_history = [0.1]
            self.metrics_history = {"synergy_shannon_entropy": [0.42]}

        def scenario_degradation(self):
            return 0.0

    dummy = DummyROITracker()
    ft.capture_from_roi(dummy, "wf")

    entry = ft.history["wf"][0]
    assert entry["synergy_shannon_entropy"] == 0.42


def test_capture_from_roi_accepts_stage_and_compute_stability():
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

    dummy.roi_history.append(1.0)
    dummy.raroi_history.append(1.0)
    dummy.confidence_history.append(0.9)
    dummy.metrics_history["synergy_resilience"].append(0.8)
    ft.capture_from_roi(dummy, "wf", stage="alpha", compute_stability=True)

    dummy.roi_history.append(2.0)
    dummy.raroi_history.append(2.0)
    dummy.confidence_history.append(0.9)
    dummy.metrics_history["synergy_resilience"].append(0.8)
    ft.capture_from_roi(dummy, "wf", stage="beta", compute_stability=True)

    entry = ft.history["wf"][-1]
    assert entry["stage"] == "beta"
    assert isinstance(entry["stability"], float)
    assert entry["stability"] > 0


def test_capture_from_roi_blends_template_and_real_roi(monkeypatch):
    def fake_load(self):
        self.templates = {"wf": [0.5, 0.5, 0.5, 0.5, 0.5]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()

    class DummyROITracker:
        def __init__(self):
            self.roi_history = []
            self.raroi_history = []
            self.confidence_history = []
            self.metrics_history = {"synergy_resilience": []}

        def scenario_degradation(self):
            return 0.0

    dummy = DummyROITracker()

    # First cycle uses template value exclusively
    dummy.roi_history.append(1.0)
    tracker.capture_from_roi(dummy, "wf")
    # Second cycle blends real ROI with template according to alpha
    dummy.roi_history.append(1.0)
    tracker.capture_from_roi(dummy, "wf")
    # Additional cycles to reach warm state where alpha == 1.0
    for _ in range(4):
        dummy.roi_history.append(1.0)
        tracker.capture_from_roi(dummy, "wf")

    history = tracker.history["wf"]
    assert history[0]["roi_delta"] == pytest.approx(0.5)
    assert history[1]["roi_delta"] == pytest.approx(0.6)
    assert history[-1]["roi_delta"] == pytest.approx(1.0)
    assert history[0]["raw_roi_delta"] == pytest.approx(1.0)


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


def test_cold_start_persists_through_serialization(monkeypatch):
    def fake_load(self):
        self.templates = {"wf": [0.5, 0.5, 0.5, 0.5, 0.5]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)

    class DummyROITracker:
        def __init__(self):
            self.roi_history: list[float] = []
            self.raroi_history: list[float] = []
            self.confidence_history: list[float] = []
            self.metrics_history = {"synergy_resilience": []}

        def scenario_degradation(self) -> float:
            return 0.0

    dummy = DummyROITracker()
    tracker = ForesightTracker()
    for _ in range(2):
        dummy.roi_history.append(1.0)
        tracker.capture_from_roi(dummy, "wf")

    assert tracker.is_cold_start("wf")

    data = tracker.to_dict()
    restored = ForesightTracker.from_dict(data)
    assert restored.is_cold_start("wf")

    dummy.roi_history.append(1.0)
    restored.capture_from_roi(dummy, "wf")
    assert restored.history["wf"][-1]["roi_delta"] == pytest.approx(0.7)


def test_get_template_value(monkeypatch):
    def fake_load(self):
        self.templates = {"wf": [0.5, 0.6]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    assert tracker.get_template_value("wf", 0) == pytest.approx(0.5)
    assert tracker.get_template_value("wf", 5) == pytest.approx(0.6)
    assert tracker.get_template_value("unknown", 0) is None


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


def test_record_cycle_metrics_accepts_extra_numeric_fields():
    tracker = ForesightTracker()
    tracker.record_cycle_metrics("wf", {"m": 1.0}, stage=1, degradation=-0.2)
    entry = tracker.history["wf"][0]
    assert entry["stage"] == 1.0
    assert entry["degradation"] == -0.2


def test_get_temporal_profile_returns_chronological_entries():
    tracker = ForesightTracker(max_cycles=3)
    tracker.record_cycle_metrics("wf", {"m": 1.0}, stage=0)
    tracker.record_cycle_metrics("wf", {"m": 2.0}, stage=1)
    profile = tracker.get_temporal_profile("wf")
    assert profile == [{"m": 1.0, "stage": 0.0}, {"m": 2.0, "stage": 1.0}]


@pytest.mark.parametrize(
    "roi_values,degradations,threshold,risk,expected_cycles,brittle",
    [
        # Stable trend with positive slope and low volatility
        ([1.0, 1.1, 1.2, 1.3], [0.0, 0.0, 0.0, 0.0], 0.5, "Stable", None, False),
        # Slow decay that reaches zero in more than two cycles
        ([1.0, 0.8, 0.6, 0.4], [0.0, 0.0, 0.0, 0.0], 5.0, "Slow decay", 4.0, False),
        # High volatility regardless of overall positive slope
        ([1.0, 4.0, 1.0, 4.0], [0.0, 0.0, 0.0, 0.0], 0.5, "Volatile", None, False),
        # Rapid collapse predicted within two cycles
        ([1.0, 0.2, 0.05], [0.0, 0.0, 0.0], 5.0, "Immediate collapse risk", 1.0, False),
        # Brittle response to minimal degradation
        ([1.0, 0.0], [0.0, 0.01], 5.0, "Immediate collapse risk", 0.0, True),
    ],
)
def test_predict_roi_collapse_scenarios(
    roi_values, degradations, threshold, risk, expected_cycles, brittle
):
    tracker = ForesightTracker(volatility_threshold=threshold)
    for roi, deg in zip(roi_values, degradations):
        tracker.record_cycle_metrics("wf", {"roi_delta": roi}, scenario_degradation=deg)
    result = tracker.predict_roi_collapse("wf")
    assert result["risk"] == risk
    if expected_cycles is None:
        assert result["collapse_in"] is None
    else:
        assert result["collapse_in"] == pytest.approx(expected_cycles)
    assert result["brittle"] is brittle


def _make_tracker(tmp_path, ent_curve):
    cfg = tmp_path / "foresight_templates.yaml"
    cfg.write_text(
        "entropy_profiles:\n  wf: base\nentropy_trajectories:\n  base:\n" +
        "".join(f"    - {v}\n" for v in ent_curve),
        encoding="utf8",
    )
    return ForesightTracker(templates_path=cfg)


@pytest.mark.parametrize(
    "roi_values,degradations,template,risk,collapse,brittle",
    [
        ([1.0, 1.1, 1.2, 1.3], [0.05] * 4, [0.05] * 5, "Stable", None, False),
        ([1.0, 0.95, 0.9, 0.85], [0.05] * 4, [0.05] * 5, "Slow decay", None, False),
        ([1.0, 3.0, 1.0, 3.0, 1.0], [0.05] * 5, [0.05] * 5, "Volatile", None, False),
        (
            [1.0, 0.5, -0.5],
            [0.05, 0.1, 0.3],
            [0.05, 0.1, 0.15],
            "Immediate collapse risk",
            0.0,
            False,
        ),
        (
            [1.0, 0.9, 0.6],
            [0.10, 0.10, 0.12],
            [0.10, 0.10, 0.11],
            "Slow decay",
            None,
            True,
        ),
    ],
)
def test_predict_roi_collapse_with_entropy_templates(
    tmp_path, roi_values, degradations, template, risk, collapse, brittle
):
    tracker = _make_tracker(tmp_path, template)
    for roi, deg in zip(roi_values, degradations):
        tracker.record_cycle_metrics(
            "wf", {"roi_delta": roi, "scenario_degradation": deg}
        )
    result = tracker.predict_roi_collapse("wf")
    assert result["risk"] == risk
    assert result["brittle"] is brittle
    if collapse is None:
        assert result["collapse_in"] is None
    else:
        assert result["collapse_in"] == pytest.approx(collapse)
