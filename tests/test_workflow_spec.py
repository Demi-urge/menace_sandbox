import json

import workflow_spec as ws
from workflow_synthesizer import save_workflow
from pathlib import Path


def test_to_spec_and_save(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKFLOW_OUTPUT_DIR", str(tmp_path))

    steps = [
        {"name": "step1", "bot": "a", "args": "x"},
        {"name": "step2", "bot": "b", "args": "y"},
    ]
    spec = ws.to_spec(steps)
    path = ws.save(spec, Path("step1.workflow.json"))
    assert path.name == "step1.workflow.json"

    data = json.loads(path.read_text())
    assert [s["module"] for s in data["steps"]] == ["step1", "step2"]
    md = data.get("metadata", {})
    assert md.get("workflow_id")
    assert md.get("created_at")
    assert md.get("parent_id") is None
    assert md.get("mutation_description") == ""


def test_save_workflow_with_parent(tmp_path):
    parent_spec = {
        "steps": [{"module": "base", "inputs": [], "outputs": []}],
        "metadata": {"workflow_id": "orig"},
    }
    # Persist parent workflow so a diff can be generated
    ws.save(parent_spec, tmp_path / "orig.workflow.json")

    steps = [{"module": "step1", "inputs": [], "outputs": []}]
    path = tmp_path / "wf.workflow.json"
    out_path, metadata = save_workflow(
        steps, path, parent_id="orig", mutation_description="tweak"
    )
    data = json.loads(out_path.read_text())
    md = data["metadata"]
    assert metadata["parent_id"] == "orig"
    assert metadata["mutation_description"] == "tweak"
    assert metadata["workflow_id"]
    assert metadata["created_at"]
    assert md == metadata
    diff_path = Path(md["diff_path"])
    assert diff_path.is_file()
    assert diff_path.name == f"{md['workflow_id']}.diff"
