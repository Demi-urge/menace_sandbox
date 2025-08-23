import menace.foresight_gate as fg
from menace.upgrade_forecaster import ForecastResult, CycleProjection


class DummyForecaster:
    def __init__(self, result, tracker=None, logger=None):
        self._result = result
        self.tracker = tracker
        self.logger = logger

    def forecast(self, workflow_id, patch, cycles=None, simulations=None):
        return self._result


class DummyLogger:
    def __init__(self):
        self.last = None

    def log(self, payload):
        self.last = payload


def test_projected_roi_below_threshold(stub_graph, stable_tracker):
    result = ForecastResult(
        projections=[CycleProjection(1, 0.2, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u1",
    )
    forecaster = DummyForecaster(result, tracker=stable_tracker)
    ok, reasons, forecast = fg.is_foresight_safe_to_promote(
        "wf", [], forecaster, stub_graph, roi_threshold=0.5
    )
    assert not ok
    assert "projected_roi_below_threshold" in reasons
    assert forecast.get("upgrade_id") == "u1"


def test_low_confidence(stub_graph, stable_tracker):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.5,
        upgrade_id="u2",
    )
    forecaster = DummyForecaster(result, tracker=stable_tracker)
    ok, reasons, forecast = fg.is_foresight_safe_to_promote("wf", [], forecaster, stub_graph)
    assert not ok
    assert "low_confidence" in reasons
    assert forecast.get("upgrade_id") == "u2"


def test_negative_dag_impact(negative_impact_graph, stable_tracker):
    result = ForecastResult(
        projections=[CycleProjection(1, 0.1, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u3",
    )
    logger = DummyLogger()
    forecaster = DummyForecaster(result, tracker=stable_tracker, logger=logger)
    ok, reasons, forecast = fg.is_foresight_safe_to_promote(
        "wf", [], forecaster, negative_impact_graph
    )
    assert not ok
    assert "negative_dag_impact" in reasons
    assert logger.last["reason_codes"] == ["negative_dag_impact"]
    assert logger.last["decision"] is False
    assert forecast.get("upgrade_id") == "u3"


def test_early_collapse_flag(stub_graph, brittle_tracker):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u4",
    )
    logger = DummyLogger()
    forecaster = DummyForecaster(result, tracker=brittle_tracker, logger=logger)
    ok, reasons, forecast = fg.is_foresight_safe_to_promote("wf", [], forecaster, stub_graph)
    assert not ok
    assert "roi_collapse_risk" in reasons
    assert logger.last["reason_codes"] == ["roi_collapse_risk"]
    assert logger.last["decision"] is False
    assert forecast.get("upgrade_id") == "u4"


def test_success_path(stub_graph, stable_tracker):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u5",
    )
    forecaster = DummyForecaster(result, tracker=stable_tracker)
    ok, reasons, forecast = fg.is_foresight_safe_to_promote("wf", [], forecaster, stub_graph)
    assert ok
    assert reasons == []
    assert forecast.get("upgrade_id") == "u5"
