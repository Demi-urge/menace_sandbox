from __future__ import annotations

"""Manage sibling workflow branches and offer automatic merging."""

import json
from pathlib import Path
from typing import Any, Dict, List
import logging

import workflow_lineage
from dynamic_path_router import resolve_path
from workflow_merger import merge_workflows, MergeConflictError

logger = logging.getLogger(__name__)


class WorkflowBranchManager:
    """Discover and merge sibling workflow branches."""

    def __init__(self, directory: str | Path = "workflows"):
        self.directory = resolve_path(directory)
        self._repo_root = resolve_path(".")

    # ------------------------------------------------------------------
    def _sibling_map(self) -> Dict[str, List[str]]:
        """Return mapping of parent_id to child workflow ids with siblings."""

        siblings: Dict[str, List[str]] = {}
        for spec in workflow_lineage.load_specs(self.directory):
            parent = spec.get("parent_id")
            wid = spec.get("workflow_id")
            if parent and wid:
                siblings.setdefault(str(parent), []).append(str(wid))
        return {p: c for p, c in siblings.items() if len(c) > 1}

    # ------------------------------------------------------------------
    def merge(self, parent_id: str | None = None) -> List[Path]:
        """Merge sibling branches for ``parent_id`` or all parents.

        Returns
        -------
        list[pathlib.Path]
            Paths to merged workflow specifications that were successfully
            created. Conflicting merges are skipped.
        """

        merged: List[Path] = []
        siblings = self._sibling_map()
        if parent_id is not None:
            siblings = {str(parent_id): siblings.get(str(parent_id), [])}

        for parent, children in siblings.items():
            if len(children) < 2:
                continue
            base = self.directory / f"{parent}.workflow.json"
            if not base.exists():
                continue

            try:
                base_spec = json.loads(base.read_text())
            except Exception:
                continue

            current = self._ensure_base_steps(base_spec, self.directory / f"{children[0]}.workflow.json")
            for child in children[1:]:
                branch_b = self._ensure_base_steps(base_spec, self.directory / f"{child}.workflow.json")
                out_name = f"{Path(current).stem}_{child}.workflow.json"
                out_path = self.directory / out_name
                try:
                    current = merge_workflows(base, current, branch_b, out_path)
                    try:
                        merged.append(current.relative_to(self._repo_root))
                    except ValueError:
                        merged.append(current)
                except MergeConflictError:
                    logger.info(
                        "merge conflict for parent %s between %s and %s", parent, Path(current).stem, child
                    )
                    break
        return merged

    # ------------------------------------------------------------------
    def _ensure_base_steps(self, base_spec: Dict[str, Any], path: Path) -> Path:
        """Ensure workflow at ``path`` contains steps from ``base_spec``."""

        try:
            data = json.loads(path.read_text())
        except Exception:
            return path

        base_steps = base_spec.get("steps") or []
        steps = data.get("steps") or []
        if isinstance(base_steps, list) and isinstance(steps, list):
            if steps[: len(base_steps)] != base_steps:
                data["steps"] = list(base_steps) + list(steps)
                path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return path


def merge_sibling_branches(
    directory: str | Path = "workflows", parent_id: str | None = None
) -> List[Path]:
    """Convenience wrapper around :class:`WorkflowBranchManager`."""

    manager = WorkflowBranchManager(directory)
    return manager.merge(parent_id)


__all__ = ["WorkflowBranchManager", "merge_sibling_branches"]

