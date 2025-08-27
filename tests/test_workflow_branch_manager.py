import json
import json
from pathlib import Path

from workflow_branch_manager import WorkflowBranchManager


def _create_workflow(path: Path, wid: str, parent_id: str | None, module_name: str) -> None:
    data = {
        "steps": [
            {
                "module": module_name,
                "inputs": [],
                "outputs": [],
                "files": [],
                "globals": [],
            }
        ],
        "metadata": {
            "workflow_id": wid,
            "parent_id": parent_id,
            "mutation_description": module_name,
            "created_at": "2023-01-01T00:00:00",
        },
    }
    path.write_text(json.dumps(data, indent=2))


def test_merge_sibling_branches(tmp_path):
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()

    base = wf_dir / "base.workflow.json"
    _create_workflow(base, "base", None, "mod_a")

    child_a = wf_dir / "a.workflow.json"
    _create_workflow(child_a, "a", "base", "mod_b")

    child_b = wf_dir / "b.workflow.json"
    _create_workflow(child_b, "b", "base", "mod_c")

    mgr = WorkflowBranchManager(wf_dir)
    merged = mgr.merge(parent_id="base")

    assert merged, "merge did not produce output"
    merged_data = json.loads(merged[-1].read_text())
    modules = {step["module"] for step in merged_data["steps"]}
    assert modules == {"mod_a", "mod_b", "mod_c"}
    assert merged_data["metadata"]["parent_id"] == "base"
