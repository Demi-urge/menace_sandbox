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


def test_sim_failure_falls_back_to_template(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("upgrade_forecaster.simulate_temporal_trajectory", boom)

    logs = []

    class DummyLogger:
        def log(self, data):
            logs.append(data)

    forecaster = UpgradeForecaster(
        tracker, records_base=tmp_path, logger=DummyLogger()
    )
    result = forecaster.forecast("wf", patch=[], cycles=3)

    assert [p.roi for p in result.projections] == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    ]
    assert result.confidence == pytest.approx(0.0)
    assert logs and logs[0]["event"] == "simulate_temporal_trajectory_failed"
    assert logs[0]["workflow_id"] == "wf"
    assert logs[0]["patch"] == []
    assert logs[0]["cycles"] == 3


def test_record_write_error_logged(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()

    logs = []

    class DummyLogger:
        def log(self, data):
            logs.append(data)

    forecaster = UpgradeForecaster(
        tracker, records_base=tmp_path, logger=DummyLogger()
    )

    def boom_open(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("upgrade_forecaster.Path.open", boom_open)

    result = forecaster.forecast("wf", patch=[], cycles=3)
    assert isinstance(result, ForecastResult)
    assert any(l["event"] == "record_write_failed" for l in logs)


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

    result = forecaster.forecast("wf", patch="patch123", cycles=3)
    files = list(tmp_path.glob("wf_*.json"))
    assert files, "forecast record not created"
    data = json.loads(files[0].read_text())
    upgrade_id = files[0].stem.split("_", 1)[1]
    assert data["workflow_id"] == "wf"
    assert data["patch"] == "patch123"
    assert data["upgrade_id"] == upgrade_id
    assert result.upgrade_id == upgrade_id
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
    expected = (3 / (3 + 1)) * (1 / (1 + np.var(rois)))
    assert result.confidence == pytest.approx(expected)

def test_non_cold_start_risk_template_blending(monkeypatch, tmp_path):
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
                    for _ in range(2)
                ]
            }

        def is_cold_start(self, _wf: str) -> bool:
            return False

        def get_trend_curve(self, _wf: str):
            return 0.0, 0.0, 1.0

        def get_temporal_profile(self, _wf: str):
            return list(self.history["wf"])

        def get_risk_template_curve(self, _wf: str):
            return [0.2, 0.4, 0.6]

    def fake_sim(*a, **k):
        return types.SimpleNamespace(
            roi_history=[0.0, 0.0, 0.0],
            metrics_history={"synergy_shannon_entropy": [], "synergy_risk_index": []},
        )

    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", fake_sim
    )
    tracker = DummyTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)
    result = forecaster.forecast("wf", patch=[], cycles=3)

    assert [p.risk for p in result.projections] == [
        pytest.approx(0.32),
        pytest.approx(0.44),
        pytest.approx(0.56),
    ]


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

def test_cold_start_confidence_variance(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.2, 0.2, 0.2]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    for _ in range(3):
        tracker.record_cycle_metrics("wf", {"roi_delta": 0.0})
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)

    def sim_low(*a, **k):
        return types.SimpleNamespace(roi_history=[0.2, 0.2, 0.2], metrics_history={})

    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", sim_low
    )
    low_conf = forecaster.forecast("wf", patch=[], cycles=3).confidence

    def sim_high(*a, **k):
        return types.SimpleNamespace(roi_history=[0.0, 1.0, -1.0], metrics_history={})

    monkeypatch.setattr(
        "upgrade_forecaster.simulate_temporal_trajectory", sim_high
    )
    high_conf = forecaster.forecast("wf", patch=[], cycles=3).confidence

    assert low_conf > high_conf



def test_confidence_reduces_with_run_variance(monkeypatch, tmp_path):
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

    tracker = DummyTracker()

    def make_sim(hists):
        it = iter(hists)

        def sim(*a, **k):
            hist = next(it)
            return types.SimpleNamespace(roi_history=hist, metrics_history={})

        return sim

    low_fn = make_sim([[0.2, 0.2, 0.2]] * 3)
    monkeypatch.setattr("upgrade_forecaster.simulate_temporal_trajectory", low_fn)
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path, simulations=3)
    low_conf = forecaster.forecast("wf", patch=[], cycles=3).confidence

    high_fn = make_sim([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    monkeypatch.setattr("upgrade_forecaster.simulate_temporal_trajectory", high_fn)
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path, simulations=3)
    high_conf = forecaster.forecast("wf", patch=[], cycles=3).confidence

    assert low_conf > high_conf


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
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path, horizon=1, simulations=1)

    r1 = forecaster.forecast("wf", patch=["p1"], cycles=1)
    files = list(tmp_path.glob("wf_*.json"))
    upgrade1_id = files[0].stem.split("_", 1)[1]
    assert r1.upgrade_id == upgrade1_id

    time.sleep(1)
    r2 = forecaster.forecast("wf", patch=["p2"], cycles=1)
    files = list(tmp_path.glob("wf_*.json"))
    assert len(files) == 2
    upgrade_ids = {f.stem.split("_", 1)[1] for f in files}
    assert upgrade1_id in upgrade_ids

    upgrade2_id = (upgrade_ids - {upgrade1_id}).pop()
    assert r2.upgrade_id == upgrade2_id

    first = load_record("wf", upgrade_id=upgrade1_id, records_base=tmp_path)
    second = load_record("wf", upgrade_id=upgrade2_id, records_base=tmp_path)
    latest = load_record("wf", records_base=tmp_path)

    assert first.projections[0].roi != second.projections[0].roi
    assert latest.projections[0].roi == pytest.approx(second.projections[0].roi)



def test_list_records(tmp_path):
    r1 = {"workflow_id": "wf", "upgrade_id": "a", "timestamp": 1}
    r2 = {"workflow_id": "wf", "upgrade_id": "b", "timestamp": 2}
    (tmp_path / "wf_a.json").write_text(json.dumps(r1))
    (tmp_path / "wf_b.json").write_text(json.dumps(r2))
    (tmp_path / "ignore.txt").write_text("not json")

    records = list_records(tmp_path)
    assert records == [r1, r2]


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

    delete_record("wf", upgrade_id=first_id, records_base=tmp_path)
    remaining = list_records(tmp_path)
    assert len(remaining) == 1
    assert first_id not in {r["upgrade_id"] for r in remaining}

    delete_record("wf", records_base=tmp_path)
    assert list_records(tmp_path) == []


def test_cssm_templates_used_when_local_missing(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {}
        self.workflow_profiles = {}
        self.entropy_templates = {}
        self.entropy_profiles = {}
        self.risk_templates = {}
        self.risk_profiles = {}

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    calls = []

    def fake_cssm(wf_id):
        calls.append(wf_id)
        return {
            "roi": [0.1, 0.2, 0.3],
            "entropy": [0.4, 0.5, 0.6],
            "risk": [0.7, 0.8, 0.9],
        }

    tracker = ForesightTracker(cssm_client=fake_cssm)
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)
    result = forecaster.forecast("wf", patch=[], cycles=3)

    assert calls == ["wf"]
    assert [p.roi for p in result.projections] == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    ]
    assert [p.decay for p in result.projections] == [
        pytest.approx(0.04),
        pytest.approx(0.10),
        pytest.approx(0.18),
    ]
    assert [p.risk for p in result.projections] == [
        pytest.approx(0.7),
        pytest.approx(0.8),
        pytest.approx(0.9),
    ]


def test_cssm_templates_cached(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {}
        self.workflow_profiles = {}

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    calls = []

    def fake_cssm(wf_id):
        calls.append(wf_id)
        return {"roi": [0.1], "entropy": [0.0], "risk": [0.5]}

    tracker = ForesightTracker(cssm_client=fake_cssm)
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path, horizon=1)

    forecaster.forecast("wf", patch=[], cycles=1)
    forecaster.forecast("wf", patch=[], cycles=1)

    assert calls == ["wf"]
