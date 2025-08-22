import sys
import types
import json
import pytest
from foresight_tracker import ForesightTracker


def _dummy_simulate(*args, **kwargs):
    class Dummy:
        roi_history = []
        metrics_history = {}

    return Dummy()


sys.modules.setdefault(
    "sandbox_runner.environment", types.SimpleNamespace(simulate_temporal_trajectory=_dummy_simulate)
)

from upgrade_forecaster import ForecastResult, UpgradeForecaster


def test_forecast_uses_template_on_cold_start(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)

    result = forecaster.forecast("wf", patch=[], cycles=3)
    assert isinstance(result, ForecastResult)
    rois = [p.roi for p in result.projections]
    assert rois == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert result.projections[0].risk == pytest.approx(0.9)
    assert result.projections[0].decay == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.0)


def test_forecast_writes_record(monkeypatch, tmp_path):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker, records_base=tmp_path)

    forecaster.forecast("wf", patch="patch123", cycles=3)
    files = list(tmp_path.iterdir())
    assert files, "forecast record not created"
    data = json.loads(files[0].read_text())
    assert data["workflow_id"] == "wf"
    assert data["patch"] == "patch123"
    assert len(data["projections"]) == 3
    assert isinstance(data["confidence"], float)

