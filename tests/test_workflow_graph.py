import sys
import types

import pytest

import menace.task_handoff_bot as thb
from menace.unified_event_bus import UnifiedEventBus
import workflow_graph as wg


def _patch_prediction_modules(monkeypatch: pytest.MonkeyPatch, start_id: str) -> None:
    """Provide lightweight substitutes for ROI and synergy helpers."""
    roi_mod = types.ModuleType("roi_predictor")

    class DummyPredictor:
        def forecast(self, history, exog=None):  # pragma: no cover - trivial
            base = float(history[-1]) if history else 0.0
            return base + 1.0, (0.0, 0.0)

    roi_mod.ROIPredictor = DummyPredictor
    monkeypatch.setitem(sys.modules, "roi_predictor", roi_mod)

    syn_tools = types.ModuleType("synergy_tools")
    monkeypatch.setitem(sys.modules, "synergy_tools", syn_tools)

    syn_mod = types.ModuleType("synergy_history_db")

    class _Conn:
        def close(self):  # pragma: no cover - trivial
            pass

    def connect(_path):
        return _Conn()

    def fetch_latest(_conn):
        return {start_id: 0.2}

    syn_mod.connect = connect
    syn_mod.fetch_latest = fetch_latest
    monkeypatch.setitem(sys.modules, "synergy_history_db", syn_mod)


@pytest.fixture
def populated_graph(tmp_path, monkeypatch):
    """Return a graph with three interconnected workflows."""
    bus = UnifiedEventBus()
    g = wg.WorkflowGraph(path=str(tmp_path / "graph.json"))
    g.attach_event_bus(bus)

    monkeypatch.setattr(thb.WorkflowDB, "_embed", lambda self, text: [])
    db = thb.WorkflowDB(tmp_path / "wf.db", event_bus=bus)
    wids = [
        db.add(thb.WorkflowRecord(workflow=[name], title=name, description=name))
        for name in ("A", "B", "C")
    ]
    a, b, c = map(str, wids)

    calls = []

    def fake_weight(src, dst):
        calls.append((src, dst))
        return 0.5

    monkeypatch.setattr(wg, "estimate_edge_weight", fake_weight)
    g.add_dependency(a, b)
    g.add_dependency(b, c)
    return g, (a, b, c), calls


def test_node_edge_weight_and_wave(populated_graph, monkeypatch):
    g, (a, b, c), calls = populated_graph

    if g._backend == "networkx":
        assert g.graph.has_node(a)
        assert g.graph.has_node(b)
        assert g.graph.has_node(c)
        assert g.graph[a][b]["impact_weight"] == 0.5
        assert g.graph[b][c]["impact_weight"] == 0.5
    else:
        assert a in g.graph["nodes"]
        assert b in g.graph["nodes"]
        assert c in g.graph["nodes"]
        assert g.graph["edges"][a][b]["impact_weight"] == 0.5
        assert g.graph["edges"][b][c]["impact_weight"] == 0.5

    assert calls == [(a, b), (b, c)]

    _patch_prediction_modules(monkeypatch, a)
    result = g.simulate_impact_wave(int(a))

    assert result[a]["roi"] == pytest.approx(1.0)
    assert result[b]["roi"] == pytest.approx(0.5)
    assert result[c]["roi"] == pytest.approx(0.25)
    assert result[a]["synergy"] == pytest.approx(0.2)
    assert result[b]["synergy"] == pytest.approx(0.1)
    assert result[c]["synergy"] == pytest.approx(0.05)
