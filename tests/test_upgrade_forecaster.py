import sys
import types
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


def test_forecast_uses_template_on_cold_start(monkeypatch):
    def fake_load(self):
        self.templates = {"wf": [0.1, 0.2, 0.3]}
        self.workflow_profiles["wf"] = "wf"

    monkeypatch.setattr(ForesightTracker, "_load_templates", fake_load)
    tracker = ForesightTracker()
    forecaster = UpgradeForecaster(tracker)

    result = forecaster.forecast("wf", patch=[], cycles=3)
    assert isinstance(result, ForecastResult)
    rois = [p.roi for p in result.projections]
    assert rois == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert result.projections[0].risk == pytest.approx(0.9)
    assert result.projections[0].decay == pytest.approx(0.0)
    assert result.confidence == pytest.approx(0.0)

