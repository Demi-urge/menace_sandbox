import menace.deployment_governance as dg
from menace.upgrade_forecaster import ForecastResult, CycleProjection
from menace.workflow_graph import WorkflowGraph
from pathlib import Path


class DummyForecaster:
    def __init__(self, result, tracker=None, logger=None):
        self._result = result
        self.tracker = tracker

    def forecast(self, workflow_id, patch, cycles=None, simulations=None):
        return self._result


class DummyTracker:
    def __init__(self, collapse):
        self._collapse = collapse

    def predict_roi_collapse(self, workflow_id):
        return self._collapse


def _make_graph(tmp_path: Path) -> WorkflowGraph:
    g = WorkflowGraph(path=str(tmp_path / "graph.json"))
    g.add_workflow("wf", roi=0.0)
    return g


def test_projected_roi_below_threshold(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 0.2, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u1",
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = _make_graph(tmp_path)
    forecaster = DummyForecaster(result, tracker=tracker)
    ok, forecast, reasons = dg.is_foresight_safe_to_promote(
        "wf", [], forecaster, graph, roi_threshold=0.5
    )
    assert not ok
    assert "projected_roi_below_threshold" in reasons
    assert forecast.upgrade_id == "u1"


def test_low_confidence(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.5,
        upgrade_id="u2",
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = _make_graph(tmp_path)
    forecaster = DummyForecaster(result, tracker=tracker)
    ok, forecast, reasons = dg.is_foresight_safe_to_promote("wf", [], forecaster, graph)
    assert not ok
    assert "low_confidence" in reasons
    assert forecast.upgrade_id == "u2"


def test_negative_dag_impact(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, -0.1, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u3",
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    graph.add_workflow("wf", roi=0.0)
    graph.add_workflow("dep", roi=0.0)
    graph.add_dependency("wf", "dep", impact_weight=1.0)
    forecaster = DummyForecaster(result, tracker=tracker)
    ok, forecast, reasons = dg.is_foresight_safe_to_promote(
        "wf", [], forecaster, graph, roi_threshold=-0.2
    )
    assert not ok
    assert "negative_dag_impact" in reasons
    assert forecast.upgrade_id == "u3"


def test_collapse_risk(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u4",
    )
    tracker = DummyTracker({"risk": "Immediate collapse risk"})
    graph = _make_graph(tmp_path)
    forecaster = DummyForecaster(result, tracker=tracker)
    ok, forecast, reasons = dg.is_foresight_safe_to_promote("wf", [], forecaster, graph)
    assert not ok
    assert "roi_collapse_risk" in reasons
    assert forecast.upgrade_id == "u4"


def test_success_path(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u5",
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = _make_graph(tmp_path)
    forecaster = DummyForecaster(result, tracker=tracker)
    ok, forecast, reasons = dg.is_foresight_safe_to_promote("wf", [], forecaster, graph)
    assert ok
    assert reasons == []
    assert forecast.upgrade_id == "u5"
