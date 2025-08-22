"""Integration tests for :mod:`workflow_graph`.

These tests exercise a tiny in-memory workflow database to ensure the
graph construction logic works without touching the heavy real database.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import workflow_graph as wg


@dataclass
class WorkflowRecord:
    workflow: list[str]
    title: str
    description: str
    wid: int = 0


class MiniWorkflowDB:
    """Very small in-memory stand-in for :class:`WorkflowDB`.

    Only the ``add`` operation is implemented which assigns a new
    workflow id to each record.
    """

    def __init__(self) -> None:
        self._next = 1

    def add(self, record: WorkflowRecord) -> int:
        record.wid = self._next
        self._next += 1
        return record.wid


@pytest.fixture
def mini_graph(tmp_path, monkeypatch):
    """Return a workflow graph populated with a tiny dependency DAG."""

    # Avoid expensive database bootstrap from existing files.
    monkeypatch.setattr(wg.WorkflowGraph, "populate_from_db", lambda self, db_path=None: None)

    graph_path: Path = tmp_path / "graph.json"
    g = wg.WorkflowGraph(path=str(graph_path))

    db = MiniWorkflowDB()
    names = ["A", "B", "C", "D"]
    wids = [str(db.add(WorkflowRecord(workflow=[n], title=n, description=n))) for n in names]
    for wid in wids:
        g.add_workflow(wid)

    calls: list[tuple[str, str]] = []

    def fake_weight(src: str, dst: str) -> float:
        calls.append((src, dst))
        return 0.5

    monkeypatch.setattr(wg, "estimate_edge_weight", fake_weight)

    # Build A -> {B, C} and B -> D : a simple DAG.
    g.add_dependency(wids[0], wids[1])
    g.add_dependency(wids[0], wids[2])
    g.add_dependency(wids[1], wids[3])

    return g, wids, calls, graph_path


def test_graph_round_trip_and_wave(mini_graph, monkeypatch):
    """Ensure DAG construction, persistence and impact propagation."""
    g, wids, calls, graph_path = mini_graph
    a, b, c, d = wids

    # Graph structure is a DAG and edge weights were requested for each edge.
    assert not g._graph_has_cycle()
    assert sorted(calls) == sorted([(a, b), (a, c), (b, d)])

    if g._backend == "networkx":
        assert g.graph.has_node(a) and g.graph.has_node(b)
        assert g.graph[a][b]["impact_weight"] == 0.5
        assert g.graph[a][c]["impact_weight"] == 0.5
        assert g.graph[b][d]["impact_weight"] == 0.5
    else:
        nodes = g.graph.get("nodes", {})
        edges = g.graph.get("edges", {})
        assert {a, b, c, d}.issubset(nodes)
        assert edges[a][b]["impact_weight"] == 0.5
        assert edges[a][c]["impact_weight"] == 0.5
        assert edges[b][d]["impact_weight"] == 0.5

    # Persist to disk and load into a fresh graph instance.
    g.save()

    monkeypatch.setattr(wg.WorkflowGraph, "populate_from_db", lambda self, db_path=None: None)
    g2 = wg.WorkflowGraph(path=str(graph_path))

    if g2._backend == "networkx":
        assert set(g2.graph.edges()) == {(a, b), (a, c), (b, d)}
    else:
        edges2 = g2.graph.get("edges", {})
        assert set(edges2.get(a, {}).keys()) == {b, c}
        assert set(edges2.get(b, {}).keys()) == {d}
        assert edges2.get(c, {}) == {}
        assert edges2.get(d, {}) == {}

    # Impact wave propagation should respect edge weights.
    result = g2.simulate_impact_wave(a, 1.0, 0.2)

    assert result[a]["roi"] == pytest.approx(1.0)
    assert result[b]["roi"] == pytest.approx(0.5)
    assert result[c]["roi"] == pytest.approx(0.5)
    assert result[d]["roi"] == pytest.approx(0.25)
    assert result[a]["synergy"] == pytest.approx(0.2)
    assert result[b]["synergy"] == pytest.approx(0.1)
    assert result[c]["synergy"] == pytest.approx(0.1)
    assert result[d]["synergy"] == pytest.approx(0.05)

