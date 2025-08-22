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
    """Persistent directed graph capturing workflow dependencies.

    Each node corresponds to a workflow and stores metrics such as ``roi`` or
    ``synergy_scores``.  A directed edge ``A -> B`` indicates that updates to
    ``A`` may influence ``B``.  Edges carry an ``impact_weight`` in ``[0, 1]``
    approximating how strongly ROI and synergy changes propagate along that
    path, plus optional metadata like ``dependency_type``.

    Methods like :meth:`simulate_impact_wave` traverse the graph and attenuate
    metric deltas through these weights so self‑improvement routines can reason
    about system‑wide ripple effects.

    Parameters
    ----------
    path:
        Location on disk where the graph is persisted. If omitted a default of
        ``sandbox_data/workflow_graph.gpickle`` is used.
    """

    def __init__(self, path: Optional[str] = None, *, db_path: Optional[str] = None) -> None:
        self.path = path or os.path.join("sandbox_data", "workflow_graph.gpickle")
        # ``_backend`` retained for compatibility with existing tests although
        # this implementation now always relies on ``networkx``.
        self._backend = "networkx" if _HAS_NX else "adjlist"
        self._lock = threading.Lock()
        self.graph = self.load()
        # Pre-populate the graph with workflows from the database if available.
        try:
            self.populate_from_db(db_path)
        except Exception:
            pass
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

    # ------------------------------------------------------------------
    # Automatic population from WorkflowDB
    # ------------------------------------------------------------------
    @staticmethod
    def _derive_edge_weights(a: "WorkflowRecord", b: "WorkflowRecord") -> tuple[float, float, float]:
        """Return resource, data and logical dependency weights between two records."""

        def _jaccard(seq1: Any, seq2: Any) -> float:
            s1, s2 = set(seq1 or []), set(seq2 or [])
            if not s1 or not s2:
                return 0.0
            inter = len(s1 & s2)
            union = len(s1 | s2)
            return inter / union if union else 0.0

        resource = _jaccard(a.assigned_bots, b.assigned_bots)

        last_step = (a.task_sequence or a.workflow or [None])[-1]
        data = 0.0
        if last_step:
            if last_step in (b.argument_strings or []):
                data = 1.0
            elif last_step in (b.task_sequence or b.workflow or []):
                data = 0.5

        logical = _jaccard(a.task_sequence or a.workflow, b.task_sequence or b.workflow)
        return resource, data, logical

    def populate_from_db(self, db_path: Optional[str] = None) -> None:
        """Populate the graph with workflows and inferred dependencies from a database."""

        try:  # Local import to avoid heavy dependency at module import time
            from task_handoff_bot import WorkflowDB, WorkflowRecord  # type: ignore
        except Exception:  # pragma: no cover - missing DB module
            return

        db = WorkflowDB(db_path) if db_path is not None else WorkflowDB()

        records: Dict[str, WorkflowRecord] = {}
        try:
            rows = db.conn.execute("SELECT * FROM workflows").fetchall()
        except Exception:
            rows = []

        for row in rows:
            try:
                rec = db._row_to_record(row)
            except Exception:
                continue
            wid = str(rec.wid)
            records[wid] = rec
            self.add_workflow(wid)

        for src_id, src_rec in records.items():
            for dst_id, dst_rec in records.items():
                if src_id == dst_id:
                    continue
                res, data, logical = self._derive_edge_weights(src_rec, dst_rec)
                if res == 0 and data == 0 and logical == 0:
                    continue
                weight = (res + data + logical) / 3.0
                try:
                    self.add_dependency(
                        src_id,
                        dst_id,
                        impact_weight=weight,
                        dependency_type="derived",
                        resource=res,
                        data=data,
                        logical=logical,
                    )
                except ValueError:
                    # Skip edges that would introduce cycles
                    continue

    def add_dependency(
        self,
        src: str,
        dst: str,
        *,
        impact_weight: Optional[float] = None,
        dependency_type: str = "default",
        **attrs: Any,
    ) -> None:
        """Add a dependency edge between workflows.

        Parameters
        ----------
        src, dst:
            Source and destination workflow ids.
        impact_weight:
            Optional explicit edge weight.  When ``None`` a heuristic weight is
            calculated via :func:`estimate_edge_weight`.
        dependency_type:
            Label describing the dependency category.
        attrs:
            Additional edge attributes to persist on the graph.
        """
        if impact_weight is None:
            impact_weight = estimate_edge_weight(src, dst)
        data: Dict[str, Any] = {
            "impact_weight": impact_weight,
            "dependency_type": dependency_type,
        }
        data.update(attrs)
        with self._lock:
            if _HAS_NX:
                self.graph.add_edge(src, dst, **data)
                if not nx.is_directed_acyclic_graph(self.graph):
                    self.graph.remove_edge(src, dst)
                    raise ValueError(f"Adding edge {src}->{dst} introduces a cycle")
            else:
                edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.setdefault(
                    "edges", {}
                )
                edges.setdefault(src, {})[dst] = data
                if self._graph_has_cycle():
                    edges[src].pop(dst, None)
                    raise ValueError(f"Adding edge {src}->{dst} introduces a cycle")
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
        # Refresh dependencies based on new workflow state.
        try:
            self.update_dependencies(workflow_id)
        except Exception:
            pass

    def _graph_has_cycle(self) -> bool:
        if _HAS_NX:
            try:
                return not nx.is_directed_acyclic_graph(self.graph)
            except Exception:
                return False
        edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.get("edges", {})
        nodes: Dict[str, Dict[str, Any]] = self.graph.get("nodes", {})
        visited: set[str] = set()
        stack: set[str] = set()

        def visit(node: str) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for nbr in edges.get(node, {}):
                if visit(nbr):
                    return True
            stack.remove(node)
            return False

        for n in nodes:
            if visit(n):
                return True
        return False

    def update_dependencies(self, workflow_id: str) -> None:
        """Recompute edges touching ``workflow_id`` while preserving DAG."""
        with self._lock:
            if _HAS_NX:
                if not self.graph.has_node(workflow_id):
                    self.graph.add_node(workflow_id)
                self.graph.remove_edges_from(list(self.graph.in_edges(workflow_id)))
                self.graph.remove_edges_from(list(self.graph.out_edges(workflow_id)))
                nodes = list(self.graph.nodes())
            else:
                nodes = list(self.graph.setdefault("nodes", {}).keys())
                edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.setdefault(
                    "edges", {}
                )
                edges.setdefault(workflow_id, {})
                for src in list(edges.keys()):
                    edges[src].pop(workflow_id, None)
                edges[workflow_id] = {}

            for node in nodes:
                if node == workflow_id:
                    continue
                w = estimate_edge_weight(workflow_id, node)
                if w > 0:
                    if _HAS_NX:
                        self.graph.add_edge(
                            workflow_id,
                            node,
                            impact_weight=w,
                            dependency_type="derived",
                        )
                        if self._graph_has_cycle():
                            self.graph.remove_edge(workflow_id, node)
                    else:
                        edges.setdefault(workflow_id, {})[node] = {
                            "impact_weight": w,
                            "dependency_type": "derived",
                        }
                        if self._graph_has_cycle():
                            edges[workflow_id].pop(node, None)
                w = estimate_edge_weight(node, workflow_id)
                if w > 0:
                    if _HAS_NX:
                        self.graph.add_edge(
                            node,
                            workflow_id,
                            impact_weight=w,
                            dependency_type="derived",
                        )
                        if self._graph_has_cycle():
                            self.graph.remove_edge(node, workflow_id)
                    else:
                        edges.setdefault(node, {})[workflow_id] = {
                            "impact_weight": w,
                            "dependency_type": "derived",
                        }
                        if self._graph_has_cycle():
                            edges[node].pop(workflow_id, None)
            self._save_unlocked()

    def simulate_impact_wave(
        self, starting_workflow_id: int
    ) -> Dict[str, Dict[str, float]]:
        """Simulate metric deltas flowing from ``starting_workflow_id``.

        A change to a workflow can ripple through the dependency graph.  This
        helper performs a topological traversal starting at ``starting_workflow_id``
        and attenuates the predicted ROI and synergy impact along outgoing
        edges using their ``impact_weight`` values.  The initial deltas for the
        starting workflow are estimated via :mod:`roi_predictor` and
        :mod:`synergy_tools` (falling back to ``0.0`` when those modules are not
        available).  For downstream nodes the deltas are purely a product of the
        propagated impact and edge weights.

        The return value maps workflow IDs to projected ``roi`` and ``synergy``
        *deltas* which downstream self‑improvement modules can consume.
        """

        # ------------------------------------------------------------------
        # Gather baseline metrics for the starting workflow using optional
        # predictor modules.  Failures are silently ignored.
        # ------------------------------------------------------------------
        start_id = str(starting_workflow_id)

        if _HAS_NX:
            start_node = self.graph.nodes.get(start_id, {})
        else:
            start_node = self.graph.get("nodes", {}).get(start_id, {})

        roi_base = float(start_node.get("roi", 0.0) or 0.0)
        roi_delta = 0.0
        try:  # pragma: no cover - optional dependency
            from roi_predictor import ROIPredictor  # type: ignore

            try:
                pred, _ = ROIPredictor().forecast([roi_base])
                roi_delta = float(pred - roi_base)
            except Exception:
                roi_delta = 0.0
        except Exception:  # pragma: no cover - predictor unavailable
            pass

        synergy_base = 0.0
        if isinstance(start_node.get("synergy_scores"), dict):
            try:
                synergy_base = float(next(iter(start_node["synergy_scores"].values())))
            except Exception:
                synergy_base = 0.0
        elif isinstance(start_node.get("synergy_scores"), (int, float)):
            synergy_base = float(start_node["synergy_scores"])

        synergy_delta = 0.0
        try:  # pragma: no cover - optional dependency
            import synergy_tools  # type: ignore  # noqa: F401  (presence only)
            from synergy_history_db import connect, fetch_latest  # type: ignore

            conn = connect(os.path.join("sandbox_data", "synergy_history.db"))
            try:
                latest = fetch_latest(conn)
            finally:
                conn.close()
            if start_id in latest:
                synergy_delta = float(latest[start_id]) - synergy_base
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Determine the set of reachable nodes and traverse them in topological
        # order.  Deltas are attenuated according to ``impact_weight``.
        # ------------------------------------------------------------------

        def _outgoing(src: str):
            if _HAS_NX:
                for dst in self.graph.successors(src):
                    data = self.graph[src][dst]
                    yield dst, float(data.get("impact_weight", 1.0) or 1.0)
            else:
                edges = self.graph.get("edges", {})
                for dst, data in edges.get(src, {}).items():
                    yield dst, float(data.get("impact_weight", 1.0) or 1.0)

        # gather reachable nodes
        reachable: set[str] = {start_id}
        stack = [start_id]
        while stack:
            node = stack.pop()
            for nbr, _w in _outgoing(node):
                if nbr not in reachable:
                    reachable.add(nbr)
                    stack.append(nbr)

        # topological order of reachable subgraph
        if _HAS_NX:
            sub = self.graph.subgraph(reachable).copy()
            order = list(nx.topological_sort(sub))
        else:
            edges = self.graph.get("edges", {})
            indeg: Dict[str, int] = {n: 0 for n in reachable}
            for src in reachable:
                for dst in edges.get(src, {}):
                    if dst in indeg:
                        indeg[dst] += 1
            q = deque([n for n, d in indeg.items() if d == 0])
            order: list[str] = []
            while q:
                n = q.popleft()
                order.append(n)
                for dst in edges.get(n, {}):
                    if dst in indeg:
                        indeg[dst] -= 1
                        if indeg[dst] == 0:
                            q.append(dst)

        impacts: Dict[str, tuple[float, float]] = {start_id: (roi_delta, synergy_delta)}
        for node in order:
            r, s = impacts.get(node, (0.0, 0.0))
            for dst, weight in _outgoing(node):
                if dst not in reachable:
                    continue
                cr, cs = impacts.get(dst, (0.0, 0.0))
                impacts[dst] = (cr + r * weight, cs + s * weight)

        return {wid: {"roi": r, "synergy": s} for wid, (r, s) in impacts.items()}

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
    def load(self, path: Optional[str] = None) -> object:  # type: ignore[override]
        """Load existing graph state from disk.

        Parameters
        ----------
        path:
            Optional location of the persisted graph.  When omitted the instance
            ``path`` attribute is used.
        Returns
        -------
        object
            Either a ``networkx.DiGraph`` or an adjacency-list dictionary.
        """

        path = path or self.path
        if os.path.exists(path):
            if _HAS_NX:
                if read_gpickle:
                    try:
                        return read_gpickle(path)
                    except Exception:
                        pass
                try:
                    with open(path, "rb") as fh:
                        return pickle.load(fh)
                except Exception:
                    pass
            else:
                with open(path, "rb") as fh:
                    try:
                        return pickle.load(fh)
                    except Exception:
                        pass
        # Default empty graph
        if _HAS_NX:
            return nx.DiGraph()
        return {"nodes": {}, "edges": {}}

    def save(self, path: Optional[str] = None) -> None:
        """Persist current graph state to disk.

        Parameters
        ----------
        path:
            Optional override for the destination path.
        """

        if path is not None:
            self.path = path
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
