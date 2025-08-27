from __future__ import annotations

"""Manage sibling workflow branches and offer automatic merging."""

from pathlib import Path
from typing import Dict, List
import logging

from . import workflow_lineage
from .workflow_merger import merge_workflows, MergeConflictError

logger = logging.getLogger(__name__)


class WorkflowBranchManager:
    """Discover and merge sibling workflow branches."""

    def __init__(self, directory: str | Path = "workflows"):
        self.directory = Path(directory)

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

            current = self.directory / f"{children[0]}.workflow.json"
            for child in children[1:]:
                branch_b = self.directory / f"{child}.workflow.json"
                out_name = f"{Path(current).stem}_{child}.workflow.json"
                out_path = self.directory / out_name
                try:
                    current = merge_workflows(base, current, branch_b, out_path)
                    merged.append(current)
                except MergeConflictError:
                    logger.info(
                        "merge conflict for parent %s between %s and %s", parent, Path(current).stem, child
                    )
                    break
        return merged


def merge_sibling_branches(
    directory: str | Path = "workflows", parent_id: str | None = None
) -> List[Path]:
    """Convenience wrapper around :class:`WorkflowBranchManager`."""

    manager = WorkflowBranchManager(directory)
    return manager.merge(parent_id)


__all__ = ["WorkflowBranchManager", "merge_sibling_branches"]

