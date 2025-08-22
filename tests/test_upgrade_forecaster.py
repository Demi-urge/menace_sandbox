import json
import sys
import time
import types

import numpy as np
import pytest
from foresight_tracker import ForesightTracker


def _dummy_simulate(*args, **kwargs):
    class Dummy:
        roi_history = []
        metrics_history = {}

    return Dummy()


env_mod = types.ModuleType("sandbox_runner.environment")
env_mod.simulate_temporal_trajectory = _dummy_simulate
pkg = types.ModuleType("sandbox_runner")
pkg.environment = env_mod
sys.modules.setdefault("sandbox_runner", pkg)
sys.modules.setdefault("sandbox_runner.environment", env_mod)

from upgrade_forecaster import (
    ForecastResult,
    UpgradeForecaster,
    delete_record,
    list_records,
    load_record,
)


def test_forecast_uses_template_on_cold_start(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"
        self.risk_templates = {"wf": [0.7, 0.8, 0.9]}
        self.risk_profiles = {"wf": "wf"}

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)

    result = forecaster.forecast("wf", patch=[], cycles=3)
    assert isinstance(result, ForecastResult)
    rois = [p.roi for p in result.projections]
    risks = [p.risk for p in result.projections]
    assert rois == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert risks == [
        pytest.approx(0.7),
        pytest.approx(0.8),
        pytest.approx(0.9),
    ]
    assert result.projections[0].decay == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.0)


def test_entropy_template_usage(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.0, 0.0, 0.0]}
        self.workflow_profiles["wf"] = "wf"
        self.entropy_templates = {"wf": [0.4, 0.5, 0.6]}
        self.entropy_profiles = {"wf": "wf"}

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)

    result = forecaster.forecast("wf", patch=[], cycles=3)
    assert [p.decay for p in result.projections] == [
        pytest.approx(0.04),
        pytest.approx(0.10),
        pytest.approx(0.18),
    ]


def test_forecast_writes_record(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)

    forecaster.forecast("wf", patch="patch123", cycles=3)
    files = list(tmp_path.glob("wf_*.json"))
    assert files, "forecast record not created"
    data = json.loads(files[0].read_text())
    patch_id = files[0].stem.split("_", 1)[1]
    assert data["workflow_id"] == "wf"
    assert data["patch"] == "patch123"
    assert data["patch_id"] == patch_id
    assert len(data["projections"]) == 3
    assert isinstance(data["confidence"], float)
    assert isinstance(data.get("timestamp"), (int, float))


def test_load_record_roundtrip(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)

    original = forecaster.forecast("wf", patch=[], cycles=3)
    loaded = load_record("wf", records_base=tmp_path)

    assert isinstance(loaded, ForecastResult)
    assert loaded == original


def test_risk_template_blending(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.1, 0.1]}
        self.workflow_profiles["wf"] = "wf"
        self.entropy_templates = {"wf": [0.4, 0.4, 0.4]}
        self.entropy_profiles = {"wf": "wf"}
        self.risk_templates = {"wf": [0.6, 0.4, 0.2]}
        self.risk_profiles = {"wf": "wf"}

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.0})
    tracker.record_cycle_metrics("wf", {"roi_delta": 0.0})

    def fake_sim(*a, **k):
        return types.SimpleNamespace(
            roi_history=[0.4, 0.5, 0.6],
            metrics_history={
                "synergy_shannon_entropy": [],
                "synergy_risk_index": [0.2, 0.4, 0.6],
            },
        )

    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", fake_sim
    )
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)
    result = forecaster.forecast("wf", patch=[], cycles=3)

    rois = [p.roi for p in result.projections]
    risks = [p.risk for p in result.projections]
    assert rois == [
        pytest.approx(0.22),
        pytest.approx(0.26),
        pytest.approx(0.30),
    ]
    assert risks == [
        pytest.approx(0.44),
        pytest.approx(0.40),
        pytest.approx(0.36),
    ]
    assert [p.decay for p in result.projections] == [
        pytest.approx(0.024),
        pytest.approx(0.048),
        pytest.approx(0.072),
    ]
    expected = (2 / (2 + 1)) * (1 / (1 + np.var(rois)))
    assert result.confidence == pytest.approx(expected)


def test_forecast_roi_risk_decay(monkeypatch, tmp_path):
    class DummyTracker:
        def __init__(self) -> None:
            self.history = {
                "wf": [
                    {
                        "roi_delta": 0.0,
                        "confidence": 1.0,
                        "resilience": 0.5,
                        "scenario_degradation": 0.0,
                    }
                    for _ in range(3)
                ]
            }

        def is_cold_start(self, _wf: str) -> bool:
            return False

        def get_trend_curve(self, _wf: str):
            return 0.0, 0.0, 1.0

        def get_temporal_profile(self, _wf: str):
            return list(self.history["wf"])

        def get_template_curve(self, _wf: str):
            return []

    def fake_sim(*a, **k):
        return types.SimpleNamespace(
            roi_history=[0.3, 0.3, 0.3],
            metrics_history={"synergy_shannon_entropy": [0.0]},
        )

    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", fake_sim
    )
    tracker = DummyTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)
    result = forecaster.forecast("wf", patch=[], cycles=3)

    rois = [p.roi for p in result.projections]
    assert rois == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert all(p.risk == pytest.approx(0.5) for p in result.projections)
    assert all(p.decay == pytest.approx(0.0) for p in result.projections)
    assert result.confidence == pytest.approx(0.75)


def test_forecast_confidence_sample_variance(monkeypatch, tmp_path):
    def make_tracker(samples: int):
        class DummyTracker:
            def __init__(self) -> None:
                self.history = {
                    "wf": [
                        {
                            "roi_delta": 0.0,
                            "confidence": 1.0,
                            "resilience": 0.5,
                            "scenario_degradation": 0.0,
                        }
                        for _ in range(samples)
                    ]
                }

            def is_cold_start(self, _wf: str) -> bool:
                return False

            def get_trend_curve(self, _wf: str):
                return 0.0, 0.0, 1.0

            def get_temporal_profile(self, _wf: str):
                return list(self.history["wf"])

        return DummyTracker()

    def make_sim(roi_hist):
        return lambda *a, **k: types.SimpleNamespace(
            roi_history=roi_hist,
            metrics_history={"synergy_shannon_entropy": [0.0]},
        )

    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", make_sim([0.2, 0.2, 0.2])
    )
    small = UpgradeForecaster(make_tracker(1), records_base=tmp_path)
    conf_small = small.forecast("wf", patch=[], cycles=3).confidence

    large = UpgradeForecaster(make_tracker(5), records_base=tmp_path)
    conf_large = large.forecast("wf", patch=[], cycles=3).confidence
    assert conf_large > conf_small

    tracker = make_tracker(3)
    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", make_sim([0.2, 0.2, 0.2])
    )
    low_var = UpgradeForecaster(tracker, records_base=tmp_path).forecast(
        "wf", patch=[], cycles=3
    ).confidence

    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", make_sim([0.0, 1.0, -1.0])
    )
    high_var = UpgradeForecaster(tracker, records_base=tmp_path).forecast(
        "wf", patch=[], cycles=3
    ).confidence
    assert low_var > high_var


def test_multiple_forecasts_coexist(monkeypatch, tmp_path):
    class DummyTracker:
        def __init__(self):
            self.history = {"wf": [{"roi_delta": 0.0}]}

        def is_cold_start(self, _wf: str) -> bool:
            return False

        def get_trend_curve(self, _wf: str):
            return 0.0, 0.0, 1.0

        def get_temporal_profile(self, _wf: str):
            return list(self.history["wf"])

        def get_template_curve(self, _wf: str):
            return []

    def sim(_wf, patch, foresight_tracker=None):
        val = 0.1 if patch == ["p1"] else 0.2
        return types.SimpleNamespace(roi_history=[val], metrics_history={})

    monkeypatch.setattr("upgrade_forecaster.simulate_temporal_trajectory", sim)
    tracker = DummyTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path, horizon=1)

    forecaster.forecast("wf", patch=["p1"], cycles=1)
    files = list(tmp_path.glob("wf_*.json"))
    patch1_id = files[0].stem.split("_", 1)[1]

    time.sleep(1)
    forecaster.forecast("wf", patch=["p2"], cycles=1)
    files = list(tmp_path.glob("wf_*.json"))
    assert len(files) == 2
    patch_ids = {f.stem.split("_", 1)[1] for f in files}
    assert patch1_id in patch_ids

    patch2_id = (patch_ids - {patch1_id}).pop()

    first = load_record("wf", patch_id=patch1_id, records_base=tmp_path)
    second = load_record("wf", patch_id=patch2_id, records_base=tmp_path)
    latest = load_record("wf", records_base=tmp_path)

    assert first.projections[0].roi == pytest.approx(0.1)
    assert second.projections[0].roi == pytest.approx(0.2)
    assert latest.projections[0].roi == pytest.approx(0.2)


def test_list_records(tmp_path):
    (tmp_path / "wf_a.json").write_text("{}")
    (tmp_path / "wf_b.json").write_text("{}")
    (tmp_path / "ignore.txt").write_text("not json")

    files = list_records(tmp_path)
    assert files == ["wf_a.json", "wf_b.json"]


def test_delete_record(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path, horizon=1)

    forecaster.forecast("wf", patch="p1", cycles=1)
    first_id = next(tmp_path.glob("wf_*.json")).stem.split("_", 1)[1]
    time.sleep(1)
    forecaster.forecast("wf", patch="p2", cycles=1)

    delete_record("wf", patch_id=first_id, records_base=tmp_path)
    remaining = list_records(tmp_path)
    assert len(remaining) == 1
    assert first_id not in remaining[0]

    delete_record("wf", records_base=tmp_path)
    assert list_records(tmp_path) == []
