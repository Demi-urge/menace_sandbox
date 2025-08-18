from __future__ import annotations

"""Utilities to reconstruct mutation lineage trees and manage branches."""

from dataclasses import dataclass
from datetime import datetime
import argparse
import json
from typing import List, Dict, Any, Optional

from .evolution_history_db import EvolutionHistoryDB
from .code_database import PatchHistoryDB, PatchRecord
from .patch_provenance import get_patch_provenance


@dataclass
class MutationLineage:
    """Reconstruct mutation trees from history databases."""

    history_db: EvolutionHistoryDB | None = None
    patch_db: PatchHistoryDB | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        if self.history_db is None:
            self.history_db = EvolutionHistoryDB()
        if self.patch_db is None:
            self.patch_db = PatchHistoryDB()

    # ------------------------------------------------------------------
    def build_tree(self, workflow_id: int) -> List[Dict[str, Any]]:
        """Return mutation tree for ``workflow_id`` including patch details."""
        tree = self.history_db.lineage_tree(workflow_id)

        def attach(node: Dict[str, Any]) -> Dict[str, Any]:
            pid = node.get("patch_id")
            if pid:
                patch = self._fetch_patch(pid)
                if patch:
                    node["patch"] = patch
            node["children"] = [attach(c) for c in node.get("children", [])]
            return node

        return [attach(n) for n in tree]

    # ------------------------------------------------------------------
    def _fetch_patch(self, patch_id: int) -> Optional[Dict[str, Any]]:
        with self.patch_db._connect() as conn:  # type: ignore[attr-defined]
            row = conn.execute(
                "SELECT id, filename, description, roi_before, roi_after, errors_before, "
                "errors_after, roi_delta, complexity_before, complexity_after, "
                "complexity_delta, predicted_roi, predicted_errors, reverted, "
                "trending_topic, ts, code_id, code_hash, source_bot, version, "
                "parent_patch_id, reason, trigger FROM patch_history WHERE id=?",
                (patch_id,),
            ).fetchone()
        if not row:
            return None
        keys = [
            "id",
            "filename",
            "description",
            "roi_before",
            "roi_after",
            "errors_before",
            "errors_after",
            "roi_delta",
            "complexity_before",
            "complexity_after",
            "complexity_delta",
            "predicted_roi",
            "predicted_errors",
            "reverted",
            "trending_topic",
            "ts",
            "code_id",
            "code_hash",
            "source_bot",
            "version",
            "parent_patch_id",
            "reason",
            "trigger",
        ]
        patch = dict(zip(keys, row))
        patch["provenance"] = get_patch_provenance(patch_id, patch_db=self.patch_db)
        return patch

    # ------------------------------------------------------------------
    def backtrack_failed_path(self, patch_id: int) -> List[int]:
        """Return lineage from ``patch_id`` up to last patch with positive ROI."""
        path: List[int] = []
        current: Optional[int] = patch_id
        with self.patch_db._connect() as conn:  # type: ignore[attr-defined]
            while current is not None:
                row = conn.execute(
                    "SELECT parent_patch_id, roi_delta FROM patch_history WHERE id=?",
                    (current,),
                ).fetchone()
                if not row:
                    break
                parent, delta = row
                path.append(current)
                if parent is None or delta > 0:
                    break
                current = parent
        return path

    # ------------------------------------------------------------------
    def clone_branch_for_ab_test(
        self,
        patch_id: int,
        description: str,
        vectors: List[tuple[str, float]] | None = None,
    ) -> int:
        """Clone a patch into a new branch for A/B testing.

        Returns the new patch id.
        """
        patch = self._fetch_patch(patch_id)
        if not patch:
            raise ValueError(f"patch {patch_id} not found")

        rec = PatchRecord(
            filename=patch["filename"],
            description=description,
            roi_before=float(patch["roi_after"]),
            roi_after=float(patch["roi_after"]),
            errors_before=int(patch["errors_after"]),
            errors_after=int(patch["errors_after"]),
            roi_delta=0.0,
            complexity_before=float(patch["complexity_after"]),
            complexity_after=float(patch["complexity_after"]),
            complexity_delta=0.0,
            predicted_roi=0.0,
            predicted_errors=0.0,
            reverted=False,
            trending_topic=patch.get("trending_topic"),
            ts=datetime.utcnow().isoformat(),
            code_id=patch.get("code_id"),
            code_hash=patch.get("code_hash"),
            source_bot=patch.get("source_bot"),
            version=patch.get("version"),
            parent_patch_id=patch_id,
            reason=None,
            trigger=None,
        )
        return self.patch_db.add(rec, vectors=vectors)


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Mutation lineage utilities")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("tree", help="print mutation tree for a workflow")
    t.add_argument("workflow_id", type=int)

    b = sub.add_parser("backtrack", help="backtrack a failing patch lineage")
    b.add_argument("patch_id", type=int)

    c = sub.add_parser("clone", help="clone a patch branch for A/B test")
    c.add_argument("patch_id", type=int)
    c.add_argument("--description", default="clone")

    args = parser.parse_args()
    lineage = MutationLineage()
    if args.cmd == "tree":
        print(json.dumps(lineage.build_tree(args.workflow_id), indent=2))
    elif args.cmd == "backtrack":
        print(json.dumps(lineage.backtrack_failed_path(args.patch_id)))
    elif args.cmd == "clone":
        print(lineage.clone_branch_for_ab_test(args.patch_id, args.description))
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
