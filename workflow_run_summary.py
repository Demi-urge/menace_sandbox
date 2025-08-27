from __future__ import annotations

"""Utilities for aggregating workflow ROI run metrics and writing summaries."""

from pathlib import Path
import json
from typing import Dict, List, Iterable

from .workflow_graph import WorkflowGraph

# In-memory ROI history per workflow
_WORKFLOW_ROI_HISTORY: Dict[str, List[float]] = {}


def record_run(workflow_id: str, roi: float) -> None:
    """Record ``roi`` for ``workflow_id``."""
    hist = _WORKFLOW_ROI_HISTORY.setdefault(str(workflow_id), [])
    hist.append(float(roi))


def save_all_summaries(directory: str | Path = ".", *, graph: WorkflowGraph | None = None) -> None:
    """Write ``{workflow_id}.summary.json`` files for recorded workflows.

    Parameters
    ----------
    directory:
        Destination directory for summary files.
    graph:
        Optional :class:`WorkflowGraph` instance used to resolve parent and
        child relationships. When omitted a new instance with default
        configuration is created.
    """
    if not _WORKFLOW_ROI_HISTORY:
        return

    graph = graph or WorkflowGraph()
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    for wid, history in _WORKFLOW_ROI_HISTORY.items():
        cumulative = float(sum(history))
        runs = len(history)
        avg = cumulative / runs if runs else 0.0
        parents: Iterable[str]
        children: Iterable[str]
        try:
            g = graph.graph
            if hasattr(g, "predecessors"):
                parents = list(g.predecessors(wid))
                children = list(g.successors(wid))
            else:  # adjacency list backend
                edges = g.get("edges", {})
                parents = [src for src, dsts in edges.items() if wid in dsts]
                children = list(edges.get(wid, {}).keys())
        except Exception:
            parents, children = [], []

        data = {
            "workflow_id": wid,
            "cumulative_roi": cumulative,
            "num_runs": runs,
            "average_roi": avg,
            "parents": list(parents),
            "children": list(children),
        }
        path = out_dir / f"{wid}.summary.json"
        path.write_text(json.dumps(data, indent=2))


def reset_history() -> None:
    """Clear stored ROI history. Primarily intended for tests."""
    _WORKFLOW_ROI_HISTORY.clear()
