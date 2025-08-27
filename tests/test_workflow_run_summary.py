import json
from menace.workflow_run_summary import record_run, save_all_summaries, reset_history
from menace.workflow_graph import WorkflowGraph
from menace.workflow_spec import save_spec


def test_summary_generation(tmp_path):
    reset_history()
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    graph.add_workflow("parent")
    graph.add_workflow("child")
    graph.add_dependency("parent", "child")

    # Create a workflow spec with a pre-existing summary path to ensure it is preserved
    spec = {"metadata": {"workflow_id": "parent", "summary_path": "old"}}
    spec_path = save_spec(spec, tmp_path / "workflows" / "parent.workflow.json")
    saved = json.loads(spec_path.read_text())
    assert saved["metadata"]["summary_path"] == "old"

    record_run("parent", 1.0)
    record_run("parent", 2.0)
    save_all_summaries(tmp_path / "workflows", graph=graph)
    data = json.loads((tmp_path / "workflows" / "parent.summary.json").read_text())
    assert data["workflow_id"] == "parent"
    assert data["cumulative_roi"] == 3.0
    assert data["num_runs"] == 2
    assert data["average_roi"] == 1.5
    assert data["parents"] == []
    assert set(data["children"]) == {"child"}

    # Ensure the workflow spec records the summary path
    spec_on_disk = json.loads((tmp_path / "workflows" / "parent.workflow.json").read_text())
    expected = str(tmp_path / "workflows" / "parent.summary.json")
    assert spec_on_disk["metadata"]["summary_path"] == expected
