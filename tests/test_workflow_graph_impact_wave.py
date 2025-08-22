import sys
import types
import sys

import pytest

from workflow_graph import WorkflowGraph


class DummyPredictor:
    def forecast(self, history, exog=None):  # pragma: no cover - trivial
        base = float(history[-1]) if history else 0.0
        return base + 10.0, (0.0, 0.0)


def _setup_modules(monkeypatch):
    roi_mod = types.ModuleType("roi_predictor")
    roi_mod.ROIPredictor = DummyPredictor
    monkeypatch.setitem(sys.modules, "roi_predictor", roi_mod)

    syn_tools = types.ModuleType("synergy_tools")
    monkeypatch.setitem(sys.modules, "synergy_tools", syn_tools)

    syn_mod = types.ModuleType("synergy_history_db")

    class _Conn:
        def close(self):
            pass

    def connect(_path):
        return _Conn()

    def fetch_latest(_conn):
        return {"A": 1.0}

    syn_mod.connect = connect
    syn_mod.fetch_latest = fetch_latest
    monkeypatch.setitem(sys.modules, "synergy_history_db", syn_mod)


@pytest.fixture(autouse=True)
def _patch_modules(monkeypatch):
    _setup_modules(monkeypatch)


def test_simulate_impact_wave(tmp_path):
    path = tmp_path / "graph.json"
    g = WorkflowGraph(path=str(path))
    g.add_workflow("A", roi=0.0)
    g.add_workflow("B", roi=0.0)
    g.add_workflow("C", roi=0.0)
    g.add_dependency("A", "B", impact_weight=0.5)
    g.add_dependency("B", "C", impact_weight=0.5)

    result = g.simulate_impact_wave("A")
    assert result["A"]["roi"] == pytest.approx(10.0)
    assert result["B"]["roi"] == pytest.approx(5.0)
    assert result["C"]["roi"] == pytest.approx(2.5)
    assert result["A"]["synergy"] == pytest.approx(1.0)
    assert result["B"]["synergy"] == pytest.approx(0.5)
    assert result["C"]["synergy"] == pytest.approx(0.25)
