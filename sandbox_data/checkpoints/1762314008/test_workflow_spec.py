import json
from pathlib import Path

import pytest

import workflow_spec as ws


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


def test_save_spec_with_parent(tmp_path):
    parent_spec = {
        "steps": [{"module": "base", "inputs": [], "outputs": []}],
        "metadata": {"workflow_id": "orig"},
    }
    ws.save(parent_spec, tmp_path / "orig.workflow.json")

    spec = {
        "steps": [{"module": "step1", "inputs": [], "outputs": []}],
        "metadata": {"parent_id": "orig", "mutation_description": "tweak"},
    }
    out_path = ws.save_spec(spec, tmp_path / "wf.workflow.json")
    data = json.loads(out_path.read_text())
    md = data["metadata"]
    assert md["parent_id"] == "orig"
    assert md["mutation_description"] == "tweak"
    assert md["workflow_id"]
    assert md["created_at"]
    diff_path = Path(md["diff_path"])
    assert diff_path.is_file()
    assert diff_path.name == f"{md['workflow_id']}.diff"


def test_validate_metadata(tmp_path):
    good = {"workflow_id": "a", "created_at": "now"}
    ws.validate_metadata(good)

    with pytest.raises(AssertionError):
        ws.validate_metadata({})


def test_workflow_id_regenerated_on_collision(tmp_path):
    spec = {"steps": [], "metadata": {"workflow_id": "dup"}}
    ws.save_spec(spec, tmp_path / "first.workflow.json")

    spec2 = {"steps": [], "metadata": {"workflow_id": "dup"}}
    path = ws.save_spec(spec2, tmp_path / "second.workflow.json")
    data = json.loads(path.read_text())
    assert data["metadata"]["workflow_id"] != "dup"


def test_workflow_id_collision_max_attempts(tmp_path, monkeypatch):
    """Raise when a unique ID cannot be generated."""

    first = ws.save_spec({"steps": []}, tmp_path / "first.workflow.json")
    existing_id = json.loads(first.read_text())["metadata"]["workflow_id"]

    # Force uuid4 to always return an existing ID
    monkeypatch.setattr(ws, "uuid4", lambda: existing_id)

    with pytest.raises(RuntimeError):
        ws.save_spec({"steps": []}, tmp_path / "second.workflow.json")


def test_parent_validation(tmp_path):
    parent_dir = tmp_path / "workflows"
    parent_dir.mkdir()
    # Missing workflow_id in metadata
    (parent_dir / "bad.workflow.json").write_text(
        json.dumps({"steps": [], "metadata": {"created_at": "now"}})
    )
    spec = {"steps": [], "metadata": {"parent_id": "bad"}}
    with pytest.raises(AssertionError):
        ws.save_spec(spec, tmp_path / "child.workflow.json")
