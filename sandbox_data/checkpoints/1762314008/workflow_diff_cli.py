#!/usr/bin/env python3
"""Print a unified diff between two workflow specifications.

The CLI locates workflow specifications saved by :mod:`workflow_spec` and
prints the diff between a parent and child workflow.  When the child workflow's
metadata references a stored diff (``diff_path``) for the given parent it is
used directly, otherwise the diff is regenerated using :mod:`difflib`.

Examples
--------
    python -m tools.workflow_diff_cli 1234 5678
    python -m tools.workflow_diff_cli parent child --dir /tmp/run
"""

from __future__ import annotations

import argparse
import json
import os
import difflib
from pathlib import Path
from typing import Tuple
from dynamic_path_router import resolve_path


def _load_spec(base: Path, workflow_id: str) -> Tuple[Path, dict]:
    """Return the path and data for ``workflow_id`` within ``base``.

    The directory is expected to contain ``*.workflow.json`` files produced by
    :func:`workflow_spec.save_spec`.
    """

    for candidate in base.glob("*.workflow.json"):
        try:
            data = json.loads(candidate.read_text())
        except Exception:
            continue
        md = data.get("metadata", {})
        if md.get("workflow_id") == workflow_id:
            return candidate, data
    raise FileNotFoundError(f"workflow {workflow_id!r} not found in {base}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show diff between two workflow specifications"
    )
    parser.add_argument("parent_id", help="ID of the parent workflow")
    parser.add_argument("child_id", help="ID of the child workflow")
    parser.add_argument(
        "--dir",
        default=os.environ.get("WORKFLOW_OUTPUT_DIR", "."),
        help="Directory containing workflow specs (defaults to WORKFLOW_OUTPUT_DIR or current directory)",
    )
    args = parser.parse_args()

    base_dir = resolve_path(args.dir)
    workflows_dir = (
        base_dir
        if base_dir.name == "workflows"
        else resolve_path("workflows", root=args.dir)
    )
    try:
        workflows_dir.relative_to(base_dir)
    except ValueError:
        workflows_dir = resolve_path(base_dir / "workflows")

    child_path, child_data = _load_spec(workflows_dir, str(args.child_id))
    md = child_data.get("metadata", {})
    diff_path = md.get("diff_path")
    if diff_path and md.get("parent_id") == str(args.parent_id):
        diff_file = Path(diff_path)
        if not diff_file.is_absolute():
            diff_file = workflows_dir / diff_file.name
        if diff_file.exists():
            print(diff_file.read_text())
            return

    parent_path, parent_data = _load_spec(workflows_dir, str(args.parent_id))

    # ``diff_path`` is added *after* the diff is generated when saving a spec;
    # remove it to regenerate the original diff.
    md.pop("diff_path", None)

    parent_lines = json.dumps(parent_data, indent=2, sort_keys=True).splitlines()
    child_lines = json.dumps(child_data, indent=2, sort_keys=True).splitlines()
    diff_lines = difflib.unified_diff(
        parent_lines,
        child_lines,
        fromfile=parent_path.name,
        tofile=child_path.name,
        lineterm="",
    )
    print("\n".join(diff_lines))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
