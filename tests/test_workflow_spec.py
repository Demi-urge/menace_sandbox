import json

import workflow_spec as ws


def test_to_spec_and_save(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKFLOW_OUTPUT_DIR", str(tmp_path))

    steps = [
        {"name": "step1", "bot": "a", "args": "x"},
        {"name": "step2", "bot": "b", "args": "y"},
    ]
    spec = ws.to_spec(steps)
    assert spec["workflow"] == ["step1", "step2"]

    path = ws.save(spec)
    assert path.name == "step1.workflow.json"
    data = json.loads(path.read_text())
    assert data["workflow"] == ["step1", "step2"]

    WorkflowDB, _ = ws._load_thb()
    db = WorkflowDB(tmp_path / "workflows.db")
    recs = db.fetch()
    assert recs and recs[0].workflow == ["step1", "step2"]
