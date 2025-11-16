from __future__ import annotations

"""Utilities for exploring evolution lineages.

This module builds a tree representation of evolution events stored in
:class:`EvolutionHistoryDB`.  It exposes simple helpers for retrieving
ancestors, descendants and computing aggregate statistics for a branch.
A small CLI is provided for ad-hoc queries.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import argparse
import json
import os
from pathlib import Path

from .evolution_history_db import EvolutionHistoryDB
try:  # optional dependency for provenance details
    from .patch_provenance import get_patch_provenance
except Exception:  # pragma: no cover - gracefully degrade
    def get_patch_provenance(patch_id: int) -> List[dict]:
        return []

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

_SUMMARY_STORE = Path(
    os.environ.get("WORKFLOW_SUMMARY_STORE", str(resolve_path("sandbox_data") / "workflows"))
)


@dataclass
class LineageStats:
    """Summary statistics for a lineage branch."""

    count: int
    avg_roi: float
    avg_performance: float


class LineageTracker:
    """High level API for lineage queries."""

    def __init__(self, db: EvolutionHistoryDB | None = None) -> None:
        self.db = db or EvolutionHistoryDB()

    # ------------------------------------------------------------------
    def build_tree(self, workflow_id: int | None = None) -> List[dict]:
        """Return the full lineage tree.

        If ``workflow_id`` is provided only that workflow is included; otherwise
        all workflows are merged into a single list of root nodes.
        """

        if workflow_id is not None:
            trees = self.db.lineage_tree(workflow_id)
        else:
            cur = self.db.conn.execute(
                "SELECT DISTINCT workflow_id FROM evolution_history WHERE workflow_id IS NOT NULL"
            )
            trees = []
            for row in cur.fetchall():
                trees.extend(self.db.lineage_tree(int(row[0])))

        # Enrich each node with information from persisted workflow summaries
        summary_cache: Dict[str, dict] = {}

        def enrich(node: dict) -> None:
            wid = node.get("workflow_id")
            if wid is not None:
                wid_str = str(wid)
                data = summary_cache.get(wid_str)
                if data is None:
                    path = _SUMMARY_STORE / f"{wid_str}.summary.json"
                    try:
                        data = json.loads(path.read_text())
                    except Exception:
                        data = {}
                    summary_cache[wid_str] = data
                node["average_roi"] = data.get("average_roi")
                node["roi_delta"] = data.get("roi_delta")
                node["mutation_description"] = data.get("mutation_description")
            for child in node.get("children", []):
                enrich(child)

        for root in trees:
            enrich(root)

        return trees

    # ------------------------------------------------------------------
    def ancestors(self, event_id: int) -> List[dict]:
        """Return ancestors of ``event_id`` from root to the node itself."""

        rows = self.db.ancestors(event_id)
        result: List[dict] = []
        for row in rows:
            result.append(
                {
                    "rowid": int(row[0]),
                    "parent_event_id": row[15],
                    "action": row[1],
                    "roi": row[4],
                    "performance": row[14],
                }
            )
        return result

    # ------------------------------------------------------------------
    def descendants(self, event_id: int) -> Optional[dict]:
        """Return the subtree rooted at ``event_id``."""

        return self.db.subtree(event_id)

    # ------------------------------------------------------------------
    def branch_performance(self, event_id: int) -> LineageStats:
        """Return aggregate ROI and performance for branch rooted at ``event_id``."""

        subtree = self.db.subtree(event_id)
        if not subtree:
            return LineageStats(count=0, avg_roi=0.0, avg_performance=0.0)

        rois: List[float] = []
        perfs: List[float] = []

        def walk(node: dict) -> None:
            rois.append(float(node.get("roi", 0.0)))
            perfs.append(float(node.get("performance", 0.0)))
            for child in node.get("children", []):
                walk(child)

        walk(subtree)
        count = len(rois)
        avg_roi = sum(rois) / count if count else 0.0
        avg_perf = sum(perfs) / count if count else 0.0
        return LineageStats(count=count, avg_roi=avg_roi, avg_performance=avg_perf)

    # ------------------------------------------------------------------
    def patch_provenance(self, patch_id: int) -> List[dict]:
        """Expose vector ancestry for the given ``patch_id``."""

        return get_patch_provenance(patch_id)


# ----------------------------------------------------------------------
# CLI helpers

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query evolution lineages")
    parser.add_argument("--db", default="evolution_history.db", help="Path to DB")
    sub = parser.add_subparsers(dest="command")

    anc = sub.add_parser("ancestors", help="List ancestors of an event")
    anc.add_argument("event_id", type=int)

    desc = sub.add_parser("descendants", help="List descendants of an event")
    desc.add_argument("event_id", type=int)

    br = sub.add_parser(
        "branch-performance", help="Show aggregated performance for a branch"
    )
    br.add_argument("event_id", type=int)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    tracker = LineageTracker(EvolutionHistoryDB(args.db))

    if args.command == "ancestors":
        for node in tracker.ancestors(args.event_id):
            print(node)
        return 0
    if args.command == "descendants":
        print(tracker.descendants(args.event_id))
        return 0
    if args.command == "branch-performance":
        stats = tracker.branch_performance(args.event_id)
        print(stats)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
