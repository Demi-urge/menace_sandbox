from __future__ import annotations

"""Utilities for aggregating workflow ROI run metrics and writing summaries."""

from pathlib import Path
import json
import os
from typing import Dict, List

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - support package and script imports
    from .workflow_graph import WorkflowGraph
    from . import workflow_spec
except ImportError:  # pragma: no cover - fallback when executed directly
    from workflow_graph import WorkflowGraph  # type: ignore
    import workflow_spec  # type: ignore

# In-memory ROI history per workflow
_WORKFLOW_ROI_HISTORY: Dict[str, List[float]] = {}


# Location of the persistent history store. Tests can override via the
# ``WORKFLOW_ROI_HISTORY_PATH`` environment variable.
_HISTORY_PATH = Path(
    os.environ.get(
        "WORKFLOW_ROI_HISTORY_PATH",
        str(resolve_path("workflow_roi_history.json")),
    )
)

# Default location for saved workflow summaries.  Individual tests can override
# via the ``WORKFLOW_SUMMARY_STORE`` environment variable.
_SUMMARY_STORE = Path(
    os.environ.get("WORKFLOW_SUMMARY_STORE", str(resolve_path("sandbox_data") / "workflows"))
)


def _load_history() -> None:
    """Load persisted ROI history into memory."""
    if not _HISTORY_PATH.exists():
        return
    try:
        data = json.loads(_HISTORY_PATH.read_text())
    except Exception:
        data = None
    if isinstance(data, dict):
        for wid, vals in data.items():
            if isinstance(vals, list):
                _WORKFLOW_ROI_HISTORY[str(wid)] = [float(v) for v in vals]


def _load_summaries() -> None:
    """Populate ROI history from existing workflow summary files.

    Each ``*.summary.json`` in the shared summary store contains cumulative
    statistics about past runs.  We approximate the run history by repeating the
    average ROI value ``num_runs`` times.  This allows new runs to build upon
    previously recorded values even if the explicit ``workflow_roi_history``
    file is missing.
    """

    if not _SUMMARY_STORE.exists():
        return

    for path in _SUMMARY_STORE.glob("*.summary.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        wid = str(data.get("workflow_id") or path.stem.replace(".summary", ""))
        runs = int(data.get("num_runs", 0))
        avg = float(data.get("average_roi", 0.0))

        hist = _WORKFLOW_ROI_HISTORY.setdefault(wid, [])
        if len(hist) < runs:
            hist.extend([avg] * (runs - len(hist)))


def _persist_history() -> None:
    """Persist the in-memory ROI history."""
    try:
        _HISTORY_PATH.write_text(json.dumps(_WORKFLOW_ROI_HISTORY, indent=2))
    except Exception:
        pass


def _merge_history_from_summary(wid: str, directory: Path) -> None:
    """Ensure in-memory ROI history includes data from existing summaries.

    Parameters
    ----------
    wid:
        Workflow identifier.
    directory:
        Directory where a summary may already exist.  The shared summary store
        is also consulted.
    """

    hist = _WORKFLOW_ROI_HISTORY.setdefault(wid, [])
    initial_len = len(hist)
    existing_runs = 0
    avg = 0.0

    # Look for summaries in both the provided directory and the shared store.
    candidates = [Path(directory), Path(directory) / "workflows", _SUMMARY_STORE]
    for base in candidates:
        path = base / f"{wid}.summary.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        runs = int(data.get("num_runs", 0))
        if runs > existing_runs:
            existing_runs = runs
            avg = float(data.get("average_roi", 0.0))

    needed = existing_runs + initial_len
    if existing_runs and len(hist) < needed:
        hist.extend([avg] * (needed - len(hist)))


# Load any previously persisted data on import.
_load_history()
_load_summaries()


def record_run(workflow_id: str, roi: float) -> None:
    """Record ``roi`` for ``workflow_id``."""
    hist = _WORKFLOW_ROI_HISTORY.setdefault(str(workflow_id), [])
    hist.append(float(roi))
    _persist_history()


def save_summary(
    workflow_id: str, directory: Path, graph: WorkflowGraph | None = None
) -> Path:
    """Write a ``<workflow_id>.summary.json`` file and return its path."""

    wid = str(workflow_id)
    _merge_history_from_summary(wid, Path(directory))
    history = _WORKFLOW_ROI_HISTORY.get(wid, [])
    cumulative = float(sum(history))
    runs = len(history)
    avg = cumulative / runs if runs else 0.0

    graph = graph or WorkflowGraph()
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

    roi_delta: float | None = None
    avg_roi_delta: float | None = None
    parent_id = metadata.get("parent_id")
    if parent_id:
        parent_summary = Path(directory) / f"{parent_id}.summary.json"
        if not parent_summary.exists():
            parent_summary = Path(directory) / "workflows" / f"{parent_id}.summary.json"
        if parent_summary.exists():
            try:
                parent_data = json.loads(parent_summary.read_text())
                parent_cum = float(parent_data.get("cumulative_roi", 0.0))
                parent_avg = float(parent_data.get("average_roi", 0.0))
                roi_delta = cumulative - parent_cum
                avg_roi_delta = avg - parent_avg
            except Exception:
                pass

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
        "roi_delta": roi_delta,
        "avg_roi_delta": avg_roi_delta,
    }

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{wid}.summary.json"
    path.write_text(json.dumps(data, indent=2))

    # Also persist a copy to the shared summary store so other components can
    # easily discover summary information without needing the original
    # directory.
    try:
        _SUMMARY_STORE.mkdir(parents=True, exist_ok=True)
        (_SUMMARY_STORE / f"{wid}.summary.json").write_text(
            json.dumps(data, indent=2)
        )
    except Exception:
        pass

    return path


def save_all_summaries(directory: str | Path = ".", *, graph: WorkflowGraph | None = None) -> None:
    """Write ``{workflow_id}.summary.json`` files for recorded workflows.

    Parameters
    ----------
    directory:
        Destination directory for summary files.
    graph:
        Optional workflow graph to derive parent/child relationships.
    """
    if not _WORKFLOW_ROI_HISTORY:
        return

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    for wid in _WORKFLOW_ROI_HISTORY:
        # Generate and save the summary
        summary_path = save_summary(wid, out_dir, graph=graph)

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
    try:
        _HISTORY_PATH.unlink()
    except FileNotFoundError:
        pass


__all__ = ["record_run", "save_summary", "save_all_summaries", "reset_history"]
