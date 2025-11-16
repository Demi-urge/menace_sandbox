import pytest

from workflow_graph import WorkflowGraph


def test_simulate_impact_wave(tmp_path):
    path = tmp_path / "graph.json"
    g = WorkflowGraph(path=str(path))
    g.add_workflow("A", roi=0.0)
    g.add_workflow("B", roi=0.0)
    g.add_workflow("C", roi=0.0)
    g.add_dependency("A", "B", impact_weight=0.5)
    g.add_dependency("B", "C", impact_weight=0.5)

    result = g.simulate_impact_wave("A", 10.0, 1.0)
    assert result["A"]["roi"] == pytest.approx(10.0)
    assert result["B"]["roi"] == pytest.approx(5.0)
    assert result["C"]["roi"] == pytest.approx(2.5)
    assert result["A"]["synergy"] == pytest.approx(1.0)
    assert result["B"]["synergy"] == pytest.approx(0.5)
    assert result["C"]["synergy"] == pytest.approx(0.25)


def test_simulate_impact_wave_with_damping(tmp_path):
    path = tmp_path / "graph.json"
    g = WorkflowGraph(path=str(path))
    g.add_workflow("A", roi=0.0)
    g.add_workflow("B", roi=0.0)
    g.add_dependency("A", "B", resource_weight=1.0)

    result = g.simulate_impact_wave("A", 10.0, 1.0, resource_damping=0.2)
    assert result["B"]["roi"] == pytest.approx(2.0)
    assert result["B"]["synergy"] == pytest.approx(0.2)
