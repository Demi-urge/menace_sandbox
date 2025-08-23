import os
from pathlib import Path

from menace.deployment_governance import is_foresight_safe_to_promote
from menace.upgrade_forecaster import ForecastResult, CycleProjection
from menace.forecast_logger import ForecastLogger
from menace.workflow_graph import WorkflowGraph


class DummyForecaster:
    def __init__(self, result, logger=None):
        self._result = result
        self.logger = logger

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
    forecaster = DummyForecaster(result, ForecastLogger(tmp_path / "log.jsonl"))
    tracker = DummyTracker({"risk": "Stable", "brittle": False})
    graph = _make_graph(tmp_path)
    ok, reasons, _ = is_foresight_safe_to_promote(
        "wf",
        [],
        forecaster=forecaster,
        tracker=tracker,
        graph=graph,
        roi_threshold=0.5,
    )
    assert not ok
    assert "projected_roi_below_threshold" in reasons


def test_low_confidence(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.5,
        upgrade_id="u2",
    )
    forecaster = DummyForecaster(result, ForecastLogger(tmp_path / "log.jsonl"))
    tracker = DummyTracker({"risk": "Stable", "brittle": False})
    graph = _make_graph(tmp_path)
    ok, reasons, _ = is_foresight_safe_to_promote(
        "wf",
        [],
        forecaster=forecaster,
        tracker=tracker,
        graph=graph,
    )
    assert not ok
    assert "low_confidence" in reasons


def test_negative_impact_wave(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, -0.1, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u3",
    )
    forecaster = DummyForecaster(result, ForecastLogger(tmp_path / "log.jsonl"))
    tracker = DummyTracker({"risk": "Stable", "brittle": False})
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    graph.add_workflow("wf", roi=0.0)
    graph.add_workflow("dep", roi=0.0)
    graph.add_dependency("wf", "dep", impact_weight=1.0)
    ok, reasons, _ = is_foresight_safe_to_promote(
        "wf",
        [],
        forecaster=forecaster,
        tracker=tracker,
        graph=graph,
        roi_threshold=-0.2,
    )
    assert not ok
    assert "negative_impact_wave" in reasons


def test_collapse_risk(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u4",
    )
    forecaster = DummyForecaster(result, ForecastLogger(tmp_path / "log.jsonl"))
    tracker = DummyTracker({"risk": "Immediate collapse risk", "brittle": False})
    graph = _make_graph(tmp_path)
    ok, reasons, _ = is_foresight_safe_to_promote(
        "wf",
        [],
        forecaster=forecaster,
        tracker=tracker,
        graph=graph,
    )
    assert not ok
    assert "roi_collapse_risk" in reasons



def test_success_path(tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u5",
    )
    forecaster = DummyForecaster(result, ForecastLogger(tmp_path / "log.jsonl"))
    tracker = DummyTracker({"risk": "Stable", "brittle": False})
    graph = _make_graph(tmp_path)
    ok, reasons, _ = is_foresight_safe_to_promote(
        "wf",
        [],
        forecaster=forecaster,
        tracker=tracker,
        graph=graph,
    )
    assert ok
    assert reasons == []
