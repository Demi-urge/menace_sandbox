from __future__ import annotations

"""Utilities to reconstruct mutation lineage trees and manage branches.

The module can also visualise lineage trees.  :meth:`MutationLineage.render_tree`
writes a workflow's mutation lineage to ``PNG`` or ``SVG`` images or a
Graphviz ``DOT`` file depending on the extension of the output path.  When
``networkx`` is installed it is used to construct the graph; otherwise a DOT
string is generated directly.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import json
from typing import List, Dict, Any, Optional

try:  # pragma: no cover - optional dependency
    import networkx as nx  # type: ignore
    from networkx.drawing.nx_pydot import to_pydot  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore
    to_pydot = None  # type: ignore
    _HAS_NX = False

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
    def render_tree(self, workflow_id: int, out_file: Path) -> None:
        """Render mutation lineage for ``workflow_id`` to ``out_file``.

        The output format is inferred from ``out_file``'s suffix and supports
        ``.png``, ``.svg`` and ``.dot``.  When :mod:`networkx` is available the
        tree is constructed using it; otherwise a Graphviz DOT string is built
        directly.  PNG/SVG rendering falls back to :mod:`pydot` when available.
        """

        tree = self.build_tree(workflow_id)
        ext = out_file.suffix.lower()

        if _HAS_NX and to_pydot:
            g = nx.DiGraph()

            def walk(node: Dict[str, Any]) -> str:
                nid = str(node.get("rowid"))
                label = (
                    node.get("patch", {}).get("description")
                    or node.get("action")
                    or nid
                )
                g.add_node(nid, label=label)
                for child in node.get("children", []):
                    cid = walk(child)
                    g.add_edge(nid, cid)
                return nid

            for root in tree:
                walk(root)

            try:  # pragma: no cover - relies on optional deps
                pdg = to_pydot(g)
                if ext == ".png":
                    pdg.write_png(str(out_file))
                    return
                if ext == ".svg":
                    pdg.write_svg(str(out_file))
                    return
                if ext == ".dot":
                    out_file.write_text(pdg.to_string())
                    return
            except Exception:
                pass  # fall back to manual DOT generation

        lines = ["digraph G {"]

        def walk_dot(node: Dict[str, Any]) -> None:
            nid = str(node.get("rowid"))
            label = (
                node.get("patch", {}).get("description")
                or node.get("action")
                or nid
            )
            lines.append(f'"{nid}" [label="{label}"];')
            for child in node.get("children", []):
                cid = str(child.get("rowid"))
                lines.append(f'"{nid}" -> "{cid}";')
                walk_dot(child)

        for root in tree:
            walk_dot(root)
        lines.append("}")
        dot = "\n".join(lines)

        if ext == ".dot":
            out_file.write_text(dot)
            return

        try:  # pragma: no cover - optional dependency
            import pydot  # type: ignore

            (graph,) = pydot.graph_from_dot_data(dot)
            if ext == ".png":
                graph.write_png(str(out_file))
            elif ext == ".svg":
                graph.write_svg(str(out_file))
            else:
                out_file.write_text(dot)
        except Exception:  # pragma: no cover - pydot missing
            out_file.write_text(dot)

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

    v = sub.add_parser(
        "visualize", help="render mutation lineage to an image or DOT file"
    )
    v.add_argument("workflow_id", type=int)
    v.add_argument("out_file", type=Path)

    args = parser.parse_args()
    lineage = MutationLineage()
    if args.cmd == "tree":
        print(json.dumps(lineage.build_tree(args.workflow_id), indent=2))
    elif args.cmd == "backtrack":
        print(json.dumps(lineage.backtrack_failed_path(args.patch_id)))
    elif args.cmd == "clone":
        print(lineage.clone_branch_for_ab_test(args.patch_id, args.description))
    elif args.cmd == "visualize":
        lineage.render_tree(args.workflow_id, args.out_file)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
