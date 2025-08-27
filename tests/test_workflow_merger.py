import json
from datetime import datetime
from pathlib import Path

from workflow_merger import merge_workflows


def test_merge_workflows_lineage_and_diff(tmp_path):
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()

    base_data = {
        "steps": [
            {"module": "mod_a", "inputs": [], "outputs": [], "files": [], "globals": []}
        ],
        "metadata": {
            "workflow_id": "base-id",
            "parent_id": None,
            "mutation_description": "base",
            "created_at": "2023-01-01T00:00:00",
        },
    }
    branch_data = {
        "steps": [
            {"module": "mod_a", "inputs": [], "outputs": [], "files": [], "globals": []},
            {"module": "mod_b", "inputs": [], "outputs": [], "files": [], "globals": []},
        ],
        "metadata": {
            "workflow_id": "branch-id",
            "parent_id": "base-id",
            "mutation_description": "branch",
            "created_at": "2023-01-02T00:00:00",
        },
    }

    base_path = workflows_dir / "base.workflow.json"
    branch_path = workflows_dir / "branch.workflow.json"
    base_path.write_text(json.dumps(base_data, indent=2))
    branch_path.write_text(json.dumps(branch_data, indent=2))

    out_path = workflows_dir / "merged.workflow.json"
    merged_path = merge_workflows(base_path, branch_path, out_path)

    merged_data = json.loads(merged_path.read_text())
    metadata = merged_data["metadata"]

    assert metadata["parent_id"] == "base-id"
    assert metadata["workflow_id"] not in {"base-id", "branch-id"}
    assert metadata["mutation_description"].startswith("Merged")
    datetime.fromisoformat(metadata["created_at"])

    diff_path = Path(metadata["diff_path"])
    diff_text = diff_path.read_text()
    assert '"module": "mod_b"' in diff_text
    assert diff_text.startswith("---")
