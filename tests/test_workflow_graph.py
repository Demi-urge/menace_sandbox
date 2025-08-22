import os

from workflow_graph import WorkflowGraph


def test_workflow_graph_persistence(tmp_path):
    path = tmp_path / "graph.gpickle"
    g = WorkflowGraph(path=str(path))
    g.add_workflow("A", roi=1.0, synergy_scores={"s": 10})
    g.add_workflow("B", roi=2.0, synergy_scores={"s": 5})
    g.add_dependency("A", "B", impact_weight=0.5, dependency_type="requires")
    g.update_workflow("A", roi=1.5)

    g2 = WorkflowGraph(path=str(path))
    if g2._backend == "networkx":
        assert g2.graph.nodes["A"]["roi"] == 1.5
        assert g2.graph["A"]["B"]["impact_weight"] == 0.5
        assert g2.graph["A"]["B"]["dependency_type"] == "requires"
    else:
        assert g2.graph["nodes"]["A"]["roi"] == 1.5
        assert g2.graph["edges"]["A"]["B"]["impact_weight"] == 0.5
        assert g2.graph["edges"]["A"]["B"]["dependency_type"] == "requires"

    g2.remove_workflow("B")
    if g2._backend == "networkx":
        assert "B" not in g2.graph
    else:
        assert "B" not in g2.graph["nodes"]
