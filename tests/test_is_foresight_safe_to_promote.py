import menace.deployment_governance as dg
from menace.upgrade_forecaster import ForecastResult, CycleProjection
from menace.workflow_graph import WorkflowGraph
from pathlib import Path


class DummyForecaster:
    def __init__(self, result, tracker=None, logger=None):
        self._result = result

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


def test_projected_roi_below_threshold(monkeypatch, tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 0.2, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u1",
    )
    monkeypatch.setattr(
        dg, "UpgradeForecaster", lambda tracker, logger=None: DummyForecaster(result)
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = _make_graph(tmp_path)
    ok, reasons, fid = dg.is_foresight_safe_to_promote(
        "wf", [], tracker, graph, roi_threshold=0.5
    )
    assert not ok
    assert "projected_roi_below_threshold" in reasons
    assert fid == "u1"


def test_low_confidence(monkeypatch, tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.5,
        upgrade_id="u2",
    )
    monkeypatch.setattr(
        dg, "UpgradeForecaster", lambda tracker, logger=None: DummyForecaster(result)
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = _make_graph(tmp_path)
    ok, reasons, fid = dg.is_foresight_safe_to_promote("wf", [], tracker, graph)
    assert not ok
    assert "low_confidence" in reasons
    assert fid == "u2"


def test_negative_dag_impact(monkeypatch, tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, -0.1, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u3",
    )
    monkeypatch.setattr(
        dg, "UpgradeForecaster", lambda tracker, logger=None: DummyForecaster(result)
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    graph.add_workflow("wf", roi=0.0)
    graph.add_workflow("dep", roi=0.0)
    graph.add_dependency("wf", "dep", impact_weight=1.0)
    ok, reasons, fid = dg.is_foresight_safe_to_promote(
        "wf", [], tracker, graph, roi_threshold=-0.2
    )
    assert not ok
    assert "negative_dag_impact" in reasons
    assert fid == "u3"


def test_collapse_risk(monkeypatch, tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u4",
    )
    monkeypatch.setattr(
        dg, "UpgradeForecaster", lambda tracker, logger=None: DummyForecaster(result)
    )
    tracker = DummyTracker({"risk": "Immediate collapse risk"})
    graph = _make_graph(tmp_path)
    ok, reasons, fid = dg.is_foresight_safe_to_promote("wf", [], tracker, graph)
    assert not ok
    assert "roi_collapse_risk" in reasons
    assert fid == "u4"


def test_success_path(monkeypatch, tmp_path):
    result = ForecastResult(
        projections=[CycleProjection(1, 1.0, 0.0, 1.0, 0.0)],
        confidence=0.9,
        upgrade_id="u5",
    )
    monkeypatch.setattr(
        dg, "UpgradeForecaster", lambda tracker, logger=None: DummyForecaster(result)
    )
    tracker = DummyTracker({"risk": "Stable"})
    graph = _make_graph(tmp_path)
    ok, reasons, fid = dg.is_foresight_safe_to_promote("wf", [], tracker, graph)
    assert ok
    assert reasons == []
    assert fid == "u5"
