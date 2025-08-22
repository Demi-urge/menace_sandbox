"""Workflow graph management with persistence support.

This module provides a :class:`WorkflowGraph` that stores workflows and their
dependencies. It prefers :mod:`networkx` for graph management but falls back to
simple adjacency lists when NetworkX isn't available.
"""

from __future__ import annotations

import os
import pickle
import math
import atexit
import threading
from collections import defaultdict, deque
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from unified_event_bus import UnifiedEventBus

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
        self._lock = threading.Lock()
        self.graph = self.load()
        atexit.register(self._shutdown_save)

    def _shutdown_save(self) -> None:  # pragma: no cover - best effort
        try:
            self.save()
        except Exception:
            pass

    def attach_event_bus(self, bus: "UnifiedEventBus") -> None:
        """Subscribe to workflow-related events on ``bus``."""

        def _get_id(event: Any) -> Optional[str]:
            if isinstance(event, dict):
                val = event.get("workflow_id") or event.get("wid") or event.get("id")
                if val is not None:
                    return str(val)
            return None

        def _on_new(_topic: str, event: object) -> None:
            wid = _get_id(event)
            if wid is not None:
                self.add_workflow(wid)

        def _on_update(_topic: str, event: object) -> None:
            if not isinstance(event, dict):
                return
            wid = _get_id(event)
            if wid is None:
                return
            self.update_workflow(
                wid,
                roi=event.get("roi"),
                synergy_scores=event.get("synergy_scores"),
            )

        def _on_remove(_topic: str, event: object) -> None:
            wid = _get_id(event)
            if wid is None:
                return
            self.remove_workflow(wid)
            self.refresh_edges()

        bus.subscribe("workflows:new", _on_new)
        bus.subscribe("workflows:update", _on_update)
        bus.subscribe("workflows:delete", _on_remove)
        bus.subscribe("workflows:refactor", _on_remove)

    # ------------------------------------------------------------------
    # Workflow operations
    # ------------------------------------------------------------------
    def add_workflow(self, workflow_id: str, *, roi: float = 0.0,
                     synergy_scores: Optional[Any] = None) -> None:
        """Add a workflow node to the graph."""
        with self._lock:
            if _HAS_NX:
                self.graph.add_node(
                    workflow_id, roi=roi, synergy_scores=synergy_scores
                )
            else:
                nodes: Dict[str, Dict[str, Any]] = self.graph.setdefault("nodes", {})
                edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.setdefault(
                    "edges", {}
                )
                nodes[workflow_id] = {"roi": roi, "synergy_scores": synergy_scores}
                edges.setdefault(workflow_id, {})
            self._save_unlocked()

    def remove_workflow(self, workflow_id: str) -> None:
        """Remove a workflow and its edges."""
        with self._lock:
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
            self._save_unlocked()

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
        with self._lock:
            if _HAS_NX:
                self.graph.add_edge(
                    src,
                    dst,
                    impact_weight=impact_weight,
                    dependency_type=dependency_type,
                )
            else:
                edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.setdefault(
                    "edges", {}
                )
                edges.setdefault(src, {})[dst] = {
                    "impact_weight": impact_weight,
                    "dependency_type": dependency_type,
                }
            self._save_unlocked()

    def update_workflow(
        self,
        workflow_id: str,
        *,
        roi: Optional[float] = None,
        synergy_scores: Optional[Any] = None,
    ) -> None:
        """Update workflow node attributes."""
        with self._lock:
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
            self._save_unlocked()

    def simulate_impact_wave(
        self,
        starting_workflow_id: str,
        roi_delta: float = 0.0,
        synergy_delta: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """Propagate ROI and synergy deltas through outgoing edges.

        The traversal performs a breadth‑first search over the graph starting
        from ``starting_workflow_id``.  Each edge carries an ``impact_weight``
        (defaulting to ``1.0``) which attenuates the deltas as they flow to the
        target node.  Baseline metrics for each workflow are gathered from
        :class:`adaptive_roi_predictor.AdaptiveROIPredictor` and
        :mod:`synergy_history_db` when available and combined with the
        propagated deltas.  The return value maps workflow IDs to the projected
        ``roi`` and ``synergy`` values.
        """

        try:  # pragma: no cover - optional dependency
            from adaptive_roi_predictor import AdaptiveROIPredictor  # type: ignore

            roi_predictor: Any | None = AdaptiveROIPredictor()
        except Exception:  # pragma: no cover - predictor unavailable
            roi_predictor = None

        try:  # pragma: no cover - optional dependency
            from synergy_history_db import connect, fetch_latest  # type: ignore

            conn = connect(os.path.join("sandbox_data", "synergy_history.db"))
            try:
                latest_synergy = fetch_latest(conn)
            finally:
                conn.close()
        except Exception:  # pragma: no cover - history unavailable
            latest_synergy = {}

        def _baseline(wid: str) -> tuple[float, float]:
            if _HAS_NX:
                node = self.graph.nodes.get(wid, {})
            else:
                node = self.graph.get("nodes", {}).get(wid, {})
            roi_base = float(node.get("roi", 0.0) or 0.0)
            if roi_predictor is not None:
                try:
                    pred, *_ = roi_predictor.predict([[roi_base]])
                    if pred and pred[-1]:
                        roi_base = float(pred[-1][0])
                except Exception:
                    pass
            synergy_base = 0.0
            if isinstance(node.get("synergy_scores"), dict):
                try:
                    synergy_base = float(next(iter(node["synergy_scores"].values())))
                except Exception:
                    synergy_base = 0.0
            elif isinstance(node.get("synergy_scores"), (int, float)):
                synergy_base = float(node["synergy_scores"])
            synergy_base = float(latest_synergy.get(str(wid), synergy_base))
            return roi_base, synergy_base

        def _outgoing(src: str):
            if _HAS_NX:
                for dst in self.graph.successors(src):
                    data = self.graph[src][dst]
                    yield dst, float(data.get("impact_weight", 1.0) or 1.0)
            else:
                edges = self.graph.get("edges", {})
                for dst, data in edges.get(src, {}).items():
                    yield dst, float(data.get("impact_weight", 1.0) or 1.0)

        impacts: Dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
        queue: deque[tuple[str, float, float]] = deque(
            [(str(starting_workflow_id), float(roi_delta), float(synergy_delta))]
        )

        while queue:
            wid, r, s = queue.popleft()
            cr, cs = impacts[wid]
            impacts[wid] = (cr + r, cs + s)
            for nbr, weight in _outgoing(wid):
                queue.append((nbr, r * weight, s * weight))

        result: Dict[str, Dict[str, float]] = {}
        for wid, (r, s) in impacts.items():
            base_roi, base_syn = _baseline(wid)
            result[wid] = {"roi": base_roi + r, "synergy": base_syn + s}
        return result

    def refresh_edges(self) -> None:
        """Recalculate impact weights for all dependency edges."""
        with self._lock:
            if _HAS_NX:
                for src, dst in list(self.graph.edges()):
                    self.graph[src][dst]["impact_weight"] = estimate_edge_weight(
                        src, dst
                    )
            else:
                edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.get("edges", {})
                for src, targets in edges.items():
                    for dst in list(targets.keys()):
                        targets[dst]["impact_weight"] = estimate_edge_weight(src, dst)
            self._save_unlocked()

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
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if _HAS_NX and write_gpickle:
            write_gpickle(self.graph, self.path)
        else:
            with open(self.path, "wb") as fh:
                pickle.dump(self.graph, fh)


__all__ = ["WorkflowGraph", "estimate_edge_weight"]
