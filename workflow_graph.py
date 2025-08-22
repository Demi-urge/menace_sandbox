"""Workflow graph management with persistence support.

This module provides a :class:`WorkflowGraph` that stores workflows and their
dependencies. It prefers :mod:`networkx` for graph management but falls back to
simple adjacency lists when NetworkX isn't available.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

try:  # pragma: no cover - exercised indirectly
    import networkx as nx  # type: ignore
    try:  # networkx <3 provides helpers at top level, >=3 in submodule
        from networkx.readwrite.gpickle import read_gpickle, write_gpickle  # type: ignore
    except Exception:  # pragma: no cover
        read_gpickle = getattr(nx, "read_gpickle", None)
        write_gpickle = getattr(nx, "write_gpickle", None)
    _HAS_NX = True
except Exception:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore
    read_gpickle = write_gpickle = None
    _HAS_NX = False


class WorkflowGraph:
    """Graph of workflows with dependency edges.

    Parameters
    ----------
    path:
        Location on disk where the graph is persisted. If omitted a default of
        ``sandbox_data/workflow_graph.gpickle`` is used.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.path.join("sandbox_data", "workflow_graph.gpickle")
        self._backend = "networkx" if _HAS_NX else "adjlist"
        self.graph = self.load()

    # ------------------------------------------------------------------
    # Workflow operations
    # ------------------------------------------------------------------
    def add_workflow(self, workflow_id: str, *, roi: float = 0.0,
                     synergy_scores: Optional[Any] = None) -> None:
        """Add a workflow node to the graph."""
        if _HAS_NX:
            self.graph.add_node(workflow_id, roi=roi, synergy_scores=synergy_scores)
        else:
            nodes: Dict[str, Dict[str, Any]] = self.graph.setdefault("nodes", {})
            edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.setdefault(
                "edges", {}
            )
            nodes[workflow_id] = {"roi": roi, "synergy_scores": synergy_scores}
            edges.setdefault(workflow_id, {})
        self.save()

    def remove_workflow(self, workflow_id: str) -> None:
        """Remove a workflow and its edges."""
        if _HAS_NX:
            if self.graph.has_node(workflow_id):
                self.graph.remove_node(workflow_id)
        else:
            nodes: Dict[str, Dict[str, Any]] = self.graph.get("nodes", {})
            edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.get("edges", {})
            nodes.pop(workflow_id, None)
            edges.pop(workflow_id, None)
            for deps in edges.values():
                deps.pop(workflow_id, None)
        self.save()

    def add_dependency(
        self,
        src: str,
        dst: str,
        *,
        impact_weight: float = 1.0,
        dependency_type: str = "default",
    ) -> None:
        """Add a dependency edge between workflows."""
        if _HAS_NX:
            self.graph.add_edge(
                src, dst, impact_weight=impact_weight, dependency_type=dependency_type
            )
        else:
            edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.setdefault(
                "edges", {}
            )
            edges.setdefault(src, {})[dst] = {
                "impact_weight": impact_weight,
                "dependency_type": dependency_type,
            }
        self.save()

    def update_workflow(
        self,
        workflow_id: str,
        *,
        roi: Optional[float] = None,
        synergy_scores: Optional[Any] = None,
    ) -> None:
        """Update workflow node attributes."""
        if _HAS_NX:
            if not self.graph.has_node(workflow_id):
                self.graph.add_node(workflow_id)
            if roi is not None:
                self.graph.nodes[workflow_id]["roi"] = roi
            if synergy_scores is not None:
                self.graph.nodes[workflow_id]["synergy_scores"] = synergy_scores
        else:
            nodes: Dict[str, Dict[str, Any]] = self.graph.setdefault("nodes", {})
            node = nodes.setdefault(workflow_id, {})
            if roi is not None:
                node["roi"] = roi
            if synergy_scores is not None:
                node["synergy_scores"] = synergy_scores
        self.save()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def load(self):  # type: ignore[override]
        """Load existing graph state from :attr:`path`.

        Returns
        -------
        object
            Either a ``networkx.DiGraph`` or an adjacency-list dictionary.
        """

        if os.path.exists(self.path):
            if _HAS_NX:
                if read_gpickle:
                    try:
                        return read_gpickle(self.path)
                    except Exception:
                        pass
                try:
                    with open(self.path, "rb") as fh:
                        return pickle.load(fh)
                except Exception:
                    pass
            else:
                with open(self.path, "rb") as fh:
                    try:
                        return pickle.load(fh)
                    except Exception:
                        pass
        # Default empty graph
        if _HAS_NX:
            return nx.DiGraph()
        return {"nodes": {}, "edges": {}}

    def save(self) -> None:
        """Persist current graph state to :attr:`path`."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if _HAS_NX and write_gpickle:
            write_gpickle(self.graph, self.path)
        else:
            with open(self.path, "wb") as fh:
                pickle.dump(self.graph, fh)


__all__ = ["WorkflowGraph"]
