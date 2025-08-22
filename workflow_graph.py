"""Workflow graph management with persistence support.

This module provides a :class:`WorkflowGraph` that stores workflows and their
dependencies. It prefers :mod:`networkx` for graph management but falls back to
simple adjacency lists when NetworkX isn't available.
"""

from __future__ import annotations

import os
import pickle
import math
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


def estimate_edge_weight(from_id: str, to_id: str) -> float:
    """Estimate a dependency weight between two workflows.

    The heuristic looks for three primary indicators of coupling:

    * **Resource overlap** – workflows that are handled by the same bots or
      queues are more likely to influence one another.  This information is
      retrieved from :class:`task_handoff_bot.WorkflowDB` when available.
    * **API/module similarity** – overlap in ``action_chains`` (or the workflow
      steps themselves) is treated as evidence of shared modules.  When the
      optional :mod:`workflow_vectorizer` is available we augment this with a
      lightweight cosine similarity between the two workflow vectors.
    * **Output coupling** – if the output of ``from_id`` appears to be consumed
      by ``to_id`` (for example the last step of ``from_id`` appearing in the
      argument list of ``to_id``) we slightly increase the weight.

    The return value is normalised to the range ``[0, 1]``.  When any of the
    supporting modules are unavailable or the workflows cannot be located the
    function falls back to ``1.0``.
    """

    try:  # Local imports are used to avoid heavy start-up costs when optional
        from task_handoff_bot import WorkflowDB  # type: ignore
    except Exception:  # pragma: no cover - WorkflowDB unavailable
        return 1.0

    try:
        db = WorkflowDB()
    except Exception:  # pragma: no cover - database could not be opened
        return 1.0

    def _fetch(wid: Any):
        """Retrieve a :class:`WorkflowRecord` for ``wid``."""

        try:
            row = db.conn.execute("SELECT * FROM workflows WHERE id=?", (int(wid),)).fetchone()
            if row:
                return db._row_to_record(row)
        except Exception:
            return None
        return None

    a = _fetch(from_id)
    b = _fetch(to_id)
    if not a or not b:
        return 1.0

    def _jaccard(seq1: Any, seq2: Any) -> float:
        s1, s2 = set(seq1 or []), set(seq2 or [])
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0

    # Resource overlap via shared bots/queues
    resource_overlap = _jaccard(a.assigned_bots, b.assigned_bots)

    # API/module overlap from action chains or workflow steps
    module_overlap = _jaccard(a.action_chains or a.workflow, b.action_chains or b.workflow)

    # Optional vector similarity using workflow_vectorizer for a softer signal
    try:  # pragma: no cover - optional dependency
        from workflow_vectorizer import WorkflowVectorizer  # type: ignore

        vec = WorkflowVectorizer().fit([a.__dict__, b.__dict__])
        v1 = vec.transform(a.__dict__)
        v2 = vec.transform(b.__dict__)
        dot = sum(x * y for x, y in zip(v1, v2))
        norm = math.sqrt(sum(x * x for x in v1)) * math.sqrt(sum(y * y for y in v2))
        if norm:
            module_overlap = max(module_overlap, dot / norm)
    except Exception:
        pass

    # Output coupling heuristic – check if the last step of ``a`` feeds ``b``
    output_coupling = 0.0
    try:
        last_step = (a.workflow or a.task_sequence or [None])[-1]
        if last_step:
            if last_step in (b.argument_strings or []):
                output_coupling = 1.0
            elif (b.workflow or b.task_sequence or []) and last_step == (b.workflow or b.task_sequence)[0]:
                output_coupling = 0.5
    except Exception:
        pass

    weight = (resource_overlap + module_overlap + output_coupling) / 3.0
    return max(0.0, min(1.0, weight))


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
        impact_weight: Optional[float] = None,
        dependency_type: str = "default",
    ) -> None:
        """Add a dependency edge between workflows.

        When ``impact_weight`` is not provided it is estimated using
        :func:`estimate_edge_weight`.
        """
        if impact_weight is None:
            impact_weight = estimate_edge_weight(src, dst)
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

    def refresh_edges(self) -> None:
        """Recalculate impact weights for all dependency edges."""
        if _HAS_NX:
            for src, dst in list(self.graph.edges()):
                self.graph[src][dst]["impact_weight"] = estimate_edge_weight(src, dst)
        else:
            edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.get("edges", {})
            for src, targets in edges.items():
                for dst in list(targets.keys()):
                    targets[dst]["impact_weight"] = estimate_edge_weight(src, dst)
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


__all__ = ["WorkflowGraph", "estimate_edge_weight"]
