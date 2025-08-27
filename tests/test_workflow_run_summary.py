import json
from menace.workflow_run_summary import record_run, save_all_summaries, reset_history
from menace.workflow_graph import WorkflowGraph


def test_summary_generation(tmp_path):
    reset_history()
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    graph.add_workflow("parent")
    graph.add_workflow("child")
    graph.add_dependency("parent", "child")
    record_run("parent", 1.0)
    record_run("parent", 2.0)
    save_all_summaries(tmp_path, graph=graph)
    data = json.loads((tmp_path / "parent.summary.json").read_text())
    assert data["workflow_id"] == "parent"
    assert data["cumulative_roi"] == 3.0
    assert data["num_runs"] == 2
    assert data["average_roi"] == 1.5
    assert data["parents"] == []
    assert set(data["children"]) == {"child"}

