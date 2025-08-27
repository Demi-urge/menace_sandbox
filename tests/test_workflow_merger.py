import json
from datetime import datetime
from pathlib import Path

from workflow_merger import merge_workflows


def test_merge_workflows_three_way(tmp_path):
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
    branch_a_step = {
        "module": "mod_b",
        "inputs": [],
        "outputs": [],
        "files": [],
        "globals": [],
    }
    branch_a_data = {
        "steps": base_data["steps"] + [branch_a_step],
        "metadata": {
            "workflow_id": "branch-a-id",
            "parent_id": "base-id",
            "mutation_description": "branch_a",
            "created_at": "2023-01-02T00:00:00",
        },
    }
    branch_b_step = {
        "module": "mod_c",
        "inputs": [],
        "outputs": [],
        "files": [],
        "globals": [],
    }
    branch_b_data = {
        "steps": base_data["steps"] + [branch_b_step],
        "metadata": {
            "workflow_id": "branch-b-id",
            "parent_id": "base-id",
            "mutation_description": "branch_b",
            "created_at": "2023-01-03T00:00:00",
        },
    }

    base_path = workflows_dir / "base.workflow.json"
    branch_a_path = workflows_dir / "branch_a.workflow.json"
    branch_b_path = workflows_dir / "branch_b.workflow.json"
    base_path.write_text(json.dumps(base_data, indent=2))
    branch_a_path.write_text(json.dumps(branch_a_data, indent=2))
    branch_b_path.write_text(json.dumps(branch_b_data, indent=2))

    out_path = workflows_dir / "merged.workflow.json"
    merged_path = merge_workflows(base_path, branch_a_path, branch_b_path, out_path)

    merged_data = json.loads(merged_path.read_text())
    metadata = merged_data["metadata"]

    assert metadata["parent_id"] == "base-id"
    assert metadata["workflow_id"] not in {"base-id", "branch-a-id", "branch-b-id"}
    assert metadata["mutation_description"] == "merge"
    datetime.fromisoformat(metadata["created_at"])
    assert metadata["created_at"].endswith("+00:00")

    modules = {step["module"] for step in merged_data["steps"]}
    assert modules == {"mod_a", "mod_b", "mod_c"}

    diff_path = Path(metadata["diff_path"])
    diff_text = diff_path.read_text()
    assert '"module": "mod_b"' in diff_text
    assert '"module": "mod_c"' in diff_text
    assert diff_text.startswith("---")

    summary_path = Path(metadata["summary_path"])
    assert summary_path.exists()
    summary_data = json.loads(summary_path.read_text())
    assert summary_data["workflow_id"] == metadata["workflow_id"]
