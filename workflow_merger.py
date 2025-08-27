from __future__ import annotations

import json
import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import workflow_spec


class MergeConflictError(RuntimeError):
    """Raised when a merge conflict cannot be automatically resolved."""


def _merge(base: Any, a: Any, b: Any, path: str = "") -> Any:
    """Recursively merge ``a`` and ``b`` with ``base`` as common ancestor.

    On conflicting changes a :class:`MergeConflictError` is raised with a path
    describing the location of the conflict and an inline diff between the
    competing values.
    """

    if isinstance(base, dict) and isinstance(a, dict) and isinstance(b, dict):
        result: dict[str, Any] = {}
        keys = set(base) | set(a) | set(b)
        for key in keys:
            sub_path = f"{path}/{key}" if path else key
            base_val = base.get(key)
            a_val = a.get(key, base_val)
            b_val = b.get(key, base_val)
            result[key] = _merge(base_val, a_val, b_val, sub_path)
        return result

    if isinstance(base, list) and isinstance(a, list) and isinstance(b, list):
        # Reconcile lists by analysing element wise changes. Elements present in
        # ``a`` or ``b`` beyond the length of ``base`` are treated as new
        # additions. If both branches modify the same element in different ways
        # a conflict is raised.
        result: list[Any] = []
        base_len = len(base)
        for idx in range(base_len):
            base_item = base[idx]
            a_item = a[idx] if idx < len(a) else base_item
            b_item = b[idx] if idx < len(b) else base_item
            if a_item == b_item:
                result.append(a_item)
            elif a_item == base_item:
                result.append(b_item)
            elif b_item == base_item:
                result.append(a_item)
            else:
                raise MergeConflictError(_format_conflict(f"{path}[{idx}]", a_item, b_item))

        # Collect new items appended in each branch
        for seq in (a[base_len:], b[base_len:]):
            for item in seq:
                if item not in result:
                    result.append(item)

        return result

    if a == b:
        return a
    if a == base:
        return b
    if b == base:
        return a

    raise MergeConflictError(_format_conflict(path, a, b))


def _format_conflict(path: str, a: Any, b: Any) -> str:
    """Return a descriptive conflict message including a diff."""

    a_lines = json.dumps(a, indent=2, sort_keys=True).splitlines()
    b_lines = json.dumps(b, indent=2, sort_keys=True).splitlines()
    diff = "\n".join(
        difflib.unified_diff(a_lines, b_lines, fromfile="branch_a", tofile="branch_b", lineterm="")
    )
    return f"Conflict at {path or '/'}:\n{diff}"


def merge_workflows(base: Path, branch_a: Path, branch_b: Path, out_path: Path) -> Path:
    """Merge ``branch_a`` and ``branch_b`` against ``base`` and write result.

    Parameters
    ----------
    base:
        Path to the JSON workflow serving as the common ancestor.
    branch_a:
        Path to the first branch to merge.
    branch_b:
        Path to the second branch to merge.
    out_path:
        Destination for the merged workflow specification. The resulting file
        will live beneath a ``workflows`` directory as enforced by
        :func:`workflow_spec.save_spec`.
    """

    base_path = Path(base)
    branch_a_path = Path(branch_a)
    branch_b_path = Path(branch_b)
    out_path = Path(out_path)

    base_spec_full = json.loads(base_path.read_text())
    a_spec_full = json.loads(branch_a_path.read_text())
    b_spec_full = json.loads(branch_b_path.read_text())

    # Compute diffs for reference (not used directly but retained for metadata)
    base_lines = json.dumps(base_spec_full, indent=2, sort_keys=True).splitlines()
    a_lines = json.dumps(a_spec_full, indent=2, sort_keys=True).splitlines()
    b_lines = json.dumps(b_spec_full, indent=2, sort_keys=True).splitlines()
    _ = list(
        difflib.unified_diff(
            base_lines,
            a_lines,
            fromfile=base_path.name,
            tofile=branch_a_path.name,
            lineterm="",
        )
    )
    _ = list(
        difflib.unified_diff(
            base_lines,
            b_lines,
            fromfile=base_path.name,
            tofile=branch_b_path.name,
            lineterm="",
        )
    )

    # Remove metadata before merging to avoid spurious conflicts
    base_spec = dict(base_spec_full)
    a_spec = dict(a_spec_full)
    b_spec = dict(b_spec_full)
    base_spec.pop("metadata", None)
    a_spec.pop("metadata", None)
    b_spec.pop("metadata", None)

    merged_spec = _merge(base_spec, a_spec, b_spec)

    ancestor_id = base_spec_full.get("metadata", {}).get("workflow_id")
    metadata = dict(merged_spec.get("metadata") or {})
    metadata.update(
        {
            "workflow_id": str(uuid4()),
            "parent_id": ancestor_id,
            "mutation_description": "merge",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    merged_spec["metadata"] = metadata
    out_file = workflow_spec.save_spec(merged_spec, out_path)

    try:
        saved = json.loads(out_file.read_text())
        metadata = saved.get("metadata", metadata)
    except Exception:  # pragma: no cover - best effort
        saved = merged_spec
        metadata = dict(metadata)

    workflow_id = metadata.get("workflow_id")
    if workflow_id:
        try:
            from menace.workflow_run_summary import save_summary as _save_summary

            summary_path = _save_summary(str(workflow_id), out_file.parent)
            metadata["summary_path"] = str(summary_path)
            saved_meta = dict(saved.get("metadata") or {})
            saved_meta["summary_path"] = str(summary_path)
            saved["metadata"] = saved_meta
            out_file.write_text(json.dumps(saved, indent=2, sort_keys=True))
        except Exception:
            pass

    return out_file
