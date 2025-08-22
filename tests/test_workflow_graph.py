import sys
from pathlib import Path

import pytest
import workflow_graph as wg

# Expose package path so ``task_handoff_bot`` can be imported relatively.
sys.path.append(str(Path(__file__).resolve().parents[2]))
import menace_sandbox.task_handoff_bot as thb
sys.modules.setdefault("task_handoff_bot", thb)
from menace_sandbox.task_handoff_bot import WorkflowDB, WorkflowRecord


def test_build_graph_and_wave(tmp_path, monkeypatch):
    """Graph building and impact propagation using a real WorkflowDB."""

    db_path = tmp_path / "workflows.db"
    graph_path = tmp_path / "graph.json"

    db = WorkflowDB(path=db_path)
    names = ["A", "B", "C", "D"]
    records = [WorkflowRecord(workflow=[n], title=n, description=n) for n in names]
    ids = [str(db.add(r)) for r in records]
    a, b, c, d = ids

    deps = {(a, b), (a, c), (b, d)}

    def fake_derive(src_rec, dst_rec):
        key = (str(src_rec.wid), str(dst_rec.wid))
        return (0.5, 0.5, 0.5) if key in deps else (0.0, 0.0, 0.0)

    monkeypatch.setattr(wg.WorkflowGraph, "_derive_edge_weights", staticmethod(fake_derive))

    wg.build_graph(db_path=str(db_path), path=str(graph_path))

    monkeypatch.setattr(wg.WorkflowGraph, "populate_from_db", lambda self, db_path=None: None)
    g = wg.WorkflowGraph(path=str(graph_path))

    assert not g._graph_has_cycle()

    if g._backend == "networkx":
        assert set(g.graph.edges()) == {(a, b), (a, c), (b, d)}
        assert g.graph[a][b]["impact_weight"] == pytest.approx(0.5)
        assert g.graph[a][c]["impact_weight"] == pytest.approx(0.5)
        assert g.graph[b][d]["impact_weight"] == pytest.approx(0.5)
    else:
        edges = g.graph.get("edges", {})
        assert set(edges.get(a, {}).keys()) == {b, c}
        assert set(edges.get(b, {}).keys()) == {d}
        assert edges[a][b]["impact_weight"] == pytest.approx(0.5)
        assert edges[a][c]["impact_weight"] == pytest.approx(0.5)
        assert edges[b][d]["impact_weight"] == pytest.approx(0.5)

    result = g.simulate_impact_wave(a, 1.0, 0.2)
    assert result[a]["roi"] == pytest.approx(1.0)
    assert result[b]["roi"] == pytest.approx(0.5)
    assert result[c]["roi"] == pytest.approx(0.5)
    assert result[d]["roi"] == pytest.approx(0.25)
    assert result[a]["synergy"] == pytest.approx(0.2)
    assert result[b]["synergy"] == pytest.approx(0.1)
    assert result[c]["synergy"] == pytest.approx(0.1)
    assert result[d]["synergy"] == pytest.approx(0.05)
