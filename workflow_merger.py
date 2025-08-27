from __future__ import annotations

import json
import difflib
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import workflow_spec


def merge_workflows(base: Path, branch: Path, out: Path) -> Path:
    """Merge two workflow specifications.

    Parameters
    ----------
    base:
        Path to the workflow JSON representing the common ancestor.
    branch:
        Path to the workflow JSON representing the changes to merge.
    out:
        Destination for the merged workflow specification.  The resulting file
        will live beneath a ``workflows`` directory as enforced by
        :func:`workflow_spec.save_spec`.
    """

    base_path = Path(base)
    branch_path = Path(branch)
    out_path = Path(out)

    base_spec = json.loads(base_path.read_text())
    branch_spec = json.loads(branch_path.read_text())

    base_lines = json.dumps(base_spec, indent=2, sort_keys=True).splitlines()
    branch_lines = json.dumps(branch_spec, indent=2, sort_keys=True).splitlines()

    diff_lines = list(
        difflib.unified_diff(
            base_lines,
            branch_lines,
            fromfile=base_path.name,
            tofile=branch_path.name,
            lineterm="",
        )
    )

    # If there are differences favour the branch specification; otherwise keep
    # the base.  ``diff_lines`` is still computed to surface the unified diff
    # in the saved metadata via :func:`workflow_spec.save_spec`.
    merged_spec = branch_spec if diff_lines else base_spec

    ancestor_id = base_spec.get("metadata", {}).get("workflow_id")
    metadata = dict(merged_spec.get("metadata") or {})
    metadata.update(
        {
            "workflow_id": str(uuid4()),
            "parent_id": ancestor_id,
            "mutation_description": f"Merged {base_path.name} and {branch_path.name}",
            "created_at": datetime.utcnow().isoformat(),
        }
    )
    merged_spec["metadata"] = metadata

    return workflow_spec.save_spec(merged_spec, out_path)
