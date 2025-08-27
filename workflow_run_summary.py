from __future__ import annotations

"""Utilities for aggregating workflow ROI run metrics and writing summaries."""

from pathlib import Path
import json
from typing import Dict, List

from .workflow_graph import WorkflowGraph
from . import workflow_spec

# In-memory ROI history per workflow
_WORKFLOW_ROI_HISTORY: Dict[str, List[float]] = {}


def record_run(workflow_id: str, roi: float) -> None:
    """Record ``roi`` for ``workflow_id``."""
    hist = _WORKFLOW_ROI_HISTORY.setdefault(str(workflow_id), [])
    hist.append(float(roi))


def save_summary(workflow_id: str, directory: Path) -> Path:
    """Write a ``<workflow_id>.summary.json`` file and return its path."""

    wid = str(workflow_id)
    history = _WORKFLOW_ROI_HISTORY.get(wid, [])
    cumulative = float(sum(history))
    runs = len(history)
    avg = cumulative / runs if runs else 0.0

    graph = WorkflowGraph()
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

    spec_path = Path(directory) / f"{wid}.workflow.json"
    if not spec_path.exists():
        spec_path = Path(directory) / "workflows" / f"{wid}.workflow.json"
    metadata: dict = {}
    if spec_path.exists():
        try:
            spec_data = json.loads(spec_path.read_text())
        except Exception:
            spec_data = None
        if isinstance(spec_data, dict):
            md = spec_data.get("metadata")
            if isinstance(md, dict):
                metadata = md

    data = {
        "workflow_id": wid,
        "cumulative_roi": cumulative,
        "num_runs": runs,
        "average_roi": avg,
        "parents": list(parents),
        "children": list(children),
        "mutation_description": metadata.get("mutation_description", ""),
        "parent_id": metadata.get("parent_id"),
        "created_at": metadata.get("created_at"),
    }

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{wid}.summary.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def save_all_summaries(directory: str | Path = ".", *, graph: WorkflowGraph | None = None) -> None:
    """Write ``{workflow_id}.summary.json`` files for recorded workflows.

    Parameters
    ----------
    directory:
        Destination directory for summary files.
    graph:
        Unused parameter retained for backwards compatibility.
    """
    if not _WORKFLOW_ROI_HISTORY:
        return

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    for wid in _WORKFLOW_ROI_HISTORY:
        # Generate and save the summary
        summary_path = save_summary(wid, out_dir)

        # Update the workflow specification with the summary path when possible
        spec_path = out_dir / f"{wid}.workflow.json"
        if not spec_path.exists():
            spec_path = out_dir / "workflows" / f"{wid}.workflow.json"
        if spec_path.exists():
            try:
                spec_data = json.loads(spec_path.read_text())
            except Exception:
                spec_data = None
            if isinstance(spec_data, dict):
                try:
                    workflow_spec.save_spec(spec_data, spec_path, summary_path=summary_path)
                except Exception:
                    pass


def reset_history() -> None:
    """Clear stored ROI history. Primarily intended for tests."""
    _WORKFLOW_ROI_HISTORY.clear()


__all__ = ["record_run", "save_summary", "save_all_summaries", "reset_history"]
