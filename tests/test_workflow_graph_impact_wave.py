import sys
import types

import pytest

from workflow_graph import WorkflowGraph


class DummyPredictor:
    def __init__(self, *a, **k):
        pass

    def predict(self, feats, horizon=None, tracker=None, actual_roi=None, actual_class=None):
        return [[10.0]], "linear", [], None


def _setup_modules(monkeypatch):
    pred_mod = types.ModuleType("adaptive_roi_predictor")
    pred_mod.AdaptiveROIPredictor = DummyPredictor
    monkeypatch.setitem(sys.modules, "adaptive_roi_predictor", pred_mod)

    syn_mod = types.ModuleType("synergy_history_db")

    class _Conn:
        def close(self):
            pass

    def connect(_path):
        return _Conn()

    def fetch_latest(_conn):
        return {"A": 1.0, "B": 0.5}

    syn_mod.connect = connect
    syn_mod.fetch_latest = fetch_latest
    monkeypatch.setitem(sys.modules, "synergy_history_db", syn_mod)


@pytest.fixture(autouse=True)
def _patch_modules(monkeypatch):
    _setup_modules(monkeypatch)


def test_simulate_impact_wave(tmp_path):
    path = tmp_path / "graph.gpickle"
    g = WorkflowGraph(path=str(path))
    g.add_workflow("A", roi=0.0)
    g.add_workflow("B", roi=0.0)
    g.add_workflow("C", roi=0.0)
    g.add_dependency("A", "B", impact_weight=0.5)
    g.add_dependency("B", "C", impact_weight=0.5)

    result = g.simulate_impact_wave("A", roi_delta=0.1, synergy_delta=0.2)
    assert result["A"]["roi"] == pytest.approx(10.1)
    assert result["B"]["roi"] == pytest.approx(10.05)
    assert result["C"]["roi"] == pytest.approx(10.025)
    assert result["A"]["synergy"] == pytest.approx(1.2)
    assert result["B"]["synergy"] == pytest.approx(0.6)
    assert result["C"]["synergy"] == pytest.approx(0.05)
