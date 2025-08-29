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
    decision = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph, roi_threshold=0.5
    )
    assert not decision.safe
    assert decision.reasons == ["projected_roi_below_threshold"]


def test_low_confidence():
    result = _make_result(1.0, confidence=0.4)
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({})
    decision = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not decision.safe
    assert decision.reasons == ["low_confidence"]


def test_roi_collapse_risk():
    result = _make_result(1.0)
    tracker = DummyTracker({"risk": "Immediate collapse risk"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({})
    decision = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not decision.safe
    assert decision.reasons == ["roi_collapse_risk"]


def test_negative_dag_impact():
    result = _make_result(1.0)
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({"dep": {"roi": -0.1}})
    decision = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not decision.safe
    assert decision.reasons == ["negative_dag_impact"]


def test_safe_path():
    result = _make_result(1.0)
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({"dep": {"roi": 0.1}})
    decision = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert decision.safe
    assert decision.reasons == []


def test_engine_downgrades_and_logs(tmp_path):
    result = _make_result(1.0, confidence=0.4, upgrade_id="u0")
    tracker = DummyTracker({"risk": "Stable"})
    forecaster = DummyForecaster(result, tracker)
    graph = DummyGraph({})
    decision = fg.is_foresight_safe_to_promote(
        "wf", "patch", forecaster, graph
    )
    assert not decision.safe
    assert decision.reasons == ["low_confidence"]

    # mimic self_improvement gating without heavy imports
    log_path = tmp_path / "gov.jsonl"

    class _ED:
        GOVERNANCE_LOG = log_path

        @staticmethod
        def append_governance_result(scorecard, vetoes, forecast, reasons):
            with open(_ED.GOVERNANCE_LOG, "w", encoding="utf-8") as fh:
                json.dump({"forecast": forecast, "reasons": reasons}, fh)

    ed = _ED
    scorecard = {}
    vetoes: list[str] = []
    verdict = "promote"
    if not decision.safe:
        verdict = decision.recommendation
    ed.append_governance_result(
        scorecard, vetoes, decision.forecast, list(decision.reasons)
    )

    assert verdict == "pilot"
    data = json.loads(log_path.read_text())
    assert data["forecast"] == decision.forecast
    assert data["reasons"] == decision.reasons
