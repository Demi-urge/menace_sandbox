import json
import subprocess
import sys
from pathlib import Path

import workflow_spec as ws


def _create_parent_child(tmp_path: Path) -> tuple[str, str, Path]:
    parent_spec = {
        "steps": [{"module": "base", "inputs": [], "outputs": []}],
        "metadata": {"workflow_id": "parent"},
    }
    ws.save_spec(parent_spec, tmp_path / "parent.workflow.json")

    child_spec = {
        "steps": [
            {"module": "base", "inputs": [], "outputs": []},
            {"module": "child", "inputs": [], "outputs": []},
        ],
        "metadata": {"parent_id": "parent"},
    }
    child_path = ws.save_spec(child_spec, tmp_path / "child.workflow.json")
    child_md = json.loads(child_path.read_text())["metadata"]
    diff_file = Path(child_md["diff_path"])
    return "parent", child_md["workflow_id"], diff_file


def test_workflow_diff_cli(tmp_path: Path):
    parent_id, child_id, diff_path = _create_parent_child(tmp_path)

    cmd = [
        sys.executable,
        "-m",
        "tools.workflow_diff_cli",
        parent_id,
        child_id,
        "--dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    expected = diff_path.read_text().strip()
    assert result.stdout.strip() == expected

    diff_path.unlink()
    regen = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert regen.stdout.strip() == expected
