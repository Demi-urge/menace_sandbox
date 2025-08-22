import sys
import types
import sys

import pytest

import menace.task_handoff_bot as thb
from menace.unified_event_bus import UnifiedEventBus
import workflow_graph as wg


def _patch_prediction_modules(monkeypatch: pytest.MonkeyPatch, start_id: str) -> None:
    """Provide lightweight substitutes for optional modules used by the graph."""

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


def test_graph_initialization_and_event_updates(tmp_path, monkeypatch):
    bus = UnifiedEventBus()
    graph_path = tmp_path / "graph.gpickle"
    g = wg.WorkflowGraph(path=str(graph_path))
    g.attach_event_bus(bus)

    monkeypatch.setattr(thb.WorkflowDB, "_embed", lambda self, text: [])
    db = thb.WorkflowDB(tmp_path / "wf.db", event_bus=bus)

    # Graph starts empty
    if g._backend == "networkx":
        assert len(g.graph) == 0
    else:
        assert g.graph["nodes"] == {}

    # Adding via WorkflowDB publishes events and populates the graph
    wid1 = db.add(thb.WorkflowRecord(workflow=["a"], title="A", description="A"))
    wid2 = db.add(thb.WorkflowRecord(workflow=["b"], title="B", description="B"))

    if g._backend == "networkx":
        assert g.graph.has_node(str(wid1)) and g.graph.has_node(str(wid2))
    else:
        assert str(wid1) in g.graph["nodes"] and str(wid2) in g.graph["nodes"]

    # Event driven update adjusts node attributes
    bus.publish("workflows:update", {"workflow_id": wid1, "roi": 2.5})
    if g._backend == "networkx":
        assert g.graph.nodes[str(wid1)]["roi"] == 2.5
    else:
        assert g.graph["nodes"][str(wid1)]["roi"] == 2.5

    # Removing through DB emits delete event removing the node
    db.remove(wid2)
    if g._backend == "networkx":
        assert not g.graph.has_node(str(wid2))
    else:
        assert str(wid2) not in g.graph["nodes"]


def test_edge_weighting_and_impact_wave(tmp_path, monkeypatch):
    bus = UnifiedEventBus()
    graph_path = tmp_path / "graph.gpickle"
    g = wg.WorkflowGraph(path=str(graph_path))
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
    assert calls == [(a, b), (b, c)]

    if g._backend == "networkx":
        assert g.graph[a][b]["impact_weight"] == 0.5
        assert g.graph[b][c]["impact_weight"] == 0.5
    else:
        assert g.graph["edges"][a][b]["impact_weight"] == 0.5
        assert g.graph["edges"][b][c]["impact_weight"] == 0.5

    _patch_prediction_modules(monkeypatch, a)
    result = g.simulate_impact_wave(int(a))

    assert result[a]["roi"] == pytest.approx(1.0)
    assert result[b]["roi"] == pytest.approx(0.5)
    assert result[c]["roi"] == pytest.approx(0.25)
    assert result[a]["synergy"] == pytest.approx(0.2)
    assert result[b]["synergy"] == pytest.approx(0.1)
    assert result[c]["synergy"] == pytest.approx(0.05)

