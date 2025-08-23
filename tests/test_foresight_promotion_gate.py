import json

import menace.foresight_gate as fg
from menace.upgrade_forecaster import ForecastResult, CycleProjection


class DummyForecaster:
    def __init__(self, result, tracker=None, logger=None):
        self._result = result
        self.tracker = tracker
        self.logger = logger

    def forecast(self, workflow_id, patch):
        return self._result


class DummyTracker:
    def __init__(self, collapse):
        self._collapse = collapse

    def predict_roi_collapse(self, workflow_id):
        return self._collapse


class DummyGraph:
    def __init__(self, impacts):
        self._impacts = impacts

    def simulate_impact_wave(self, workflow_id, roi_delta, synergy_delta):
        return self._impacts

def _make_result(roi, confidence=0.9, decay=0.0, upgrade_id="u"):  # helper
    proj = [CycleProjection(1, roi, 0.0, 1.0, decay)]
    return ForecastResult(projections=proj, confidence=confidence, upgrade_id=upgrade_id)


def test_projected_roi_below_threshold():
    result = _make_result(0.2)
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({})
    safe, reasons, forecast = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph, roi_threshold=0.5
    )
    assert not safe
    assert reasons == ["projected_roi_below_threshold"]


def test_low_confidence():
    result = _make_result(1.0, confidence=0.4)
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({})
    safe, reasons, forecast = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not safe
    assert reasons == ["low_confidence"]


def test_roi_collapse_risk():
    result = _make_result(1.0)
    tracker = DummyTracker({"risk": "Immediate collapse risk"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({})
    safe, reasons, forecast = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not safe
    assert reasons == ["roi_collapse_risk"]


def test_negative_dag_impact():
    result = _make_result(1.0)
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({"dep": {"roi": -0.1}})
    safe, reasons, forecast = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not safe
    assert reasons == ["negative_dag_impact"]


def test_safe_path():
    result = _make_result(1.0)
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({"dep": {"roi": 0.1}})
    safe, reasons, forecast = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert safe
    assert reasons == []


def test_engine_downgrades_and_logs(tmp_path):
    result = _make_result(1.0, confidence=0.4, upgrade_id="u0")
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({})
    safe, reasons, forecast_info = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not safe
    assert reasons == ["low_confidence"]

    # mimic self_improvement_engine gating
    from menace import evaluation_dashboard as ed
    log_path = tmp_path / "gov.jsonl"
    ed.GOVERNANCE_LOG = log_path
    scorecard = {}
    vetoes: list[str] = []
    verdict = "promote"
    if not safe:
        verdict = "pilot"
    ed.append_governance_result(scorecard, vetoes, forecast_info, list(reasons))

    assert verdict == "pilot"
    data = json.loads(log_path.read_text())
    assert data["forecast"] == forecast_info
    assert data["reasons"] == reasons
