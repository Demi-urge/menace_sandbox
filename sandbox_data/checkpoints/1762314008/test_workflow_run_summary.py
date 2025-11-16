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
    spec = {
        "metadata": {
            "workflow_id": "parent",
            "summary_path": "old",
            "mutation_description": "test mutation",
            "parent_id": "root",
            "created_at": "2023-01-02T00:00:00+00:00",
        }
    }
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
    assert data["mutation_description"] == "test mutation"
    assert data["parent_id"] == "root"
    assert data["created_at"] == "2023-01-02T00:00:00+00:00"

    # Ensure the workflow spec records the summary path
    spec_on_disk = json.loads((tmp_path / "workflows" / "parent.workflow.json").read_text())
    expected = str(tmp_path / "workflows" / "parent.summary.json")
    assert spec_on_disk["metadata"]["summary_path"] == expected


def test_summary_without_spec(tmp_path):
    reset_history()
    record_run("ghost", 1.0)
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    save_all_summaries(tmp_path / "workflows", graph=graph)
    data = json.loads((tmp_path / "workflows" / "ghost.summary.json").read_text())
    assert data["mutation_description"] == ""
    assert data["parent_id"] is None
    assert data["created_at"] is None


def test_summary_with_malformed_spec(tmp_path):
    reset_history()
    spec_path = tmp_path / "workflows" / "bad.workflow.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("not json")
    record_run("bad", 2.0)
    graph = WorkflowGraph(path=str(tmp_path / "graph.json"))
    save_all_summaries(tmp_path / "workflows", graph=graph)
    data = json.loads((tmp_path / "workflows" / "bad.summary.json").read_text())
    assert data["mutation_description"] == ""
    assert data["parent_id"] is None
    assert data["created_at"] is None
