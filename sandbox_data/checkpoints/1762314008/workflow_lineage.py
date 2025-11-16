from __future__ import annotations

"""Build lineage trees for workflow specifications.

The module loads ``*.workflow.json`` files from a directory and links parent
and child workflows together.  When :mod:`networkx` is installed a
:class:`~networkx.DiGraph` is produced, otherwise a minimal adjacency mapping
is returned.  Nodes include optional ROI summary information loaded from
``{workflow_id}.summary.json`` when present.

A small CLI is provided for dumping the lineage as JSON or a Graphviz DOT
string::

    $ python workflow_lineage.py json
    $ python workflow_lineage.py graphviz
"""

from pathlib import Path
import argparse
import json
from typing import Any, Dict, Iterable, Iterator

from dynamic_path_router import resolve_path

try:  # pragma: no cover - exercised in tests via fallback
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - fallback when networkx missing
    nx = None  # type: ignore
    _HAS_NX = False


# ---------------------------------------------------------------------------
def log_lineage(
    parent_id: str | None,
    child_id: str,
    mutation_description: str | None = None,
    *,
    roi: float | None = None,
    directory: str | Path = "workflows",
) -> None:
    """Persist a lineage record for a workflow mutation.

    The function writes a ``*.workflow.json`` file describing the relationship
    between ``parent_id`` and ``child_id`` alongside an optional
    ``*.summary.json`` containing the latest ROI.  This mirrors the format used
    by :func:`load_specs` so newly logged mutations immediately become part of
    the lineage graph when that loader is invoked.

    Parameters
    ----------
    parent_id:
        Identifier of the parent workflow.  ``None`` indicates that the child
        is a root entry.
    child_id:
        Identifier for the new workflow variant.
    mutation_description:
        Optional human readable description of the mutation.
    roi:
        Optional ROI value recorded for ``child_id``.
    directory:
        Directory where the lineage specification files should be written.
    """

    directory_path = Path(directory)
    if not directory_path.is_absolute():
        directory_path = Path(resolve_path(".")) / directory_path
    directory_path.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {"workflow_id": str(child_id)}
    if parent_id is not None:
        metadata["parent_id"] = str(parent_id)
    if mutation_description is not None:
        metadata["mutation_description"] = mutation_description

    spec_path = directory_path / f"{child_id}.workflow.json"
    spec_path.write_text(json.dumps({"metadata": metadata}))

    if roi is not None:
        summary_path = directory_path / f"{child_id}.summary.json"
        summary_path.write_text(json.dumps({"roi": float(roi)}))


# ---------------------------------------------------------------------------
def _load_summary(directory: str | Path, wid: str) -> Dict[str, Any] | None:
    """Return summary data for ``wid`` if available."""
    directory_path = Path(directory)
    if not directory_path.is_absolute():
        directory_path = Path(resolve_path(".")) / directory_path
    path = directory_path / f"{wid}.summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_specs(directory: str | Path = "workflows") -> Iterator[Dict[str, Any]]:
    """Yield workflow specification metadata from ``directory``.

    Malformed files or those missing a ``workflow_id`` are ignored.  Each
    workflow's summary is loaded when available and ``roi_delta`` is computed
    relative to its ``parent_id`` when both contain ROI information.
    """

    directory_path = Path(directory)
    if not directory_path.is_absolute():
        directory_path = Path(resolve_path(".")) / directory_path
    specs: Dict[str, Dict[str, Any]] = {}

    for path in directory_path.glob("*.workflow.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        metadata = data.get("metadata") or {}
        wid = metadata.get("workflow_id")
        if not wid:
            continue
        wid_str = str(wid)
        specs[wid_str] = {
            "workflow_id": wid_str,
            "parent_id": metadata.get("parent_id"),
            "mutation_description": metadata.get("mutation_description"),
            "summary": _load_summary(path.parent, wid_str),
        }

    for spec in specs.values():
        parent = spec.get("parent_id")
        roi = None
        summary = spec.get("summary") or {}
        try:
            roi = float(summary.get("roi"))
        except Exception:
            roi = None

        parent_roi = None
        if parent:
            parent_spec = specs.get(str(parent))
            if parent_spec:
                parent_summary = parent_spec.get("summary")
            else:
                parent_summary = _load_summary(directory_path, str(parent))
            try:
                parent_roi = float(parent_summary.get("roi")) if parent_summary else None
            except Exception:
                parent_roi = None

        if roi is not None and parent_roi is not None:
            spec["roi_delta"] = roi - parent_roi
        else:
            spec["roi_delta"] = None

    for spec in specs.values():
        yield spec


def build_graph(specs: Iterable[Dict[str, Any]]) -> Any:
    """Construct and return a graph representing ``specs``.

    Returns a :class:`networkx.DiGraph` when possible, otherwise a mapping of
    ``{"nodes": {id: attrs}, "edges": {parent: [child, ...]}}``.
    """
    if _HAS_NX:
        g = nx.DiGraph()
        for spec in specs:
            wid = spec["workflow_id"]
            g.add_node(wid, **{k: v for k, v in spec.items() if k != "workflow_id"})
        for spec in specs:
            parent = spec.get("parent_id")
            wid = spec["workflow_id"]
            if parent:
                g.add_node(str(parent))
                g.add_edge(str(parent), wid)
        return g

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, list[str]] = {}
    for spec in specs:
        wid = spec["workflow_id"]
        nodes[wid] = {k: v for k, v in spec.items() if k != "workflow_id"}
    for spec in specs:
        parent = spec.get("parent_id")
        wid = spec["workflow_id"]
        if parent:
            edges.setdefault(str(parent), []).append(wid)
            nodes.setdefault(str(parent), {})
    return {"nodes": nodes, "edges": edges}


def to_json(graph: Any) -> Dict[str, Any]:
    """Return a JSON-serialisable representation of ``graph``."""
    if _HAS_NX and hasattr(graph, "nodes"):
        data = {
            "nodes": [
                {"id": n, **attrs} for n, attrs in graph.nodes(data=True)
            ],
            "edges": [
                {"source": src, "target": dst} for src, dst in graph.edges()
            ],
        }
        return data
    return graph


def to_graphviz(graph: Any) -> str:
    """Render ``graph`` as a Graphviz DOT string."""
    lines = ["digraph G {"]
    if _HAS_NX and hasattr(graph, "nodes"):
        for n, attrs in graph.nodes(data=True):
            label = attrs.get("mutation_description") or n
            attr_parts = [f'label="{label}"']
            if attrs.get("roi_delta") is not None:
                attr_parts.append(f'roi_delta="{attrs["roi_delta"]}"')
            lines.append(f'"{n}" [{", ".join(attr_parts)}];')
        for src, dst in graph.edges():
            lines.append(f'"{src}" -> "{dst}";')
    else:
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})
        for n, attrs in nodes.items():
            label = attrs.get("mutation_description") or n
            attr_parts = [f'label="{label}"']
            if attrs.get("roi_delta") is not None:
                attr_parts.append(f'roi_delta="{attrs["roi_delta"]}"')
            lines.append(f'"{n}" [{", ".join(attr_parts)}];')
        for src, dsts in edges.items():
            for dst in dsts:
                lines.append(f'"{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Workflow lineage utilities")
    parser.add_argument("--dir", default="workflows", help="spec directory")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("json", help="output lineage as JSON")
    sub.add_parser("graphviz", help="output lineage as Graphviz DOT")

    args = parser.parse_args()
    specs = list(load_specs(args.dir))
    graph = build_graph(specs)

    if args.cmd == "graphviz":
        print(to_graphviz(graph))
    else:
        print(json.dumps(to_json(graph), indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
