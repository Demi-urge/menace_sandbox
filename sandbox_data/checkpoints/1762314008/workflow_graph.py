"""Workflow graph management with persistence support.

The module exposes :class:`WorkflowGraph`, a lightweight manager for
workflow relationships.  Nodes represent individual workflows and directed
edges capture how they influence one another via an ``impact_weight`` and
optional metadata.

Key APIs
=======

``WorkflowGraph`` instances can be populated incrementally using
``add_workflow`` and ``add_dependency`` or pre-loaded from the optional
``task_handoff_bot.WorkflowDB``.  The graph state is persisted to
``sandbox_data/workflow_graph.json`` using :mod:`networkx`'s node-link format
when available and falls back to simple adjacency lists otherwise.

Once built, :meth:`simulate_impact_wave` projects ROI and synergy deltas
through the DAG so downstream systems can reason about ripple effects.
``attach_event_bus`` hooks the graph up to a
:class:`~unified_event_bus.UnifiedEventBus` so other components can publish
``workflows:*`` events and keep the structure in sync.
"""

from __future__ import annotations

import os
import json
import math
import atexit
import threading
from datetime import datetime
from collections import deque
from typing import Any, Dict, Optional, TYPE_CHECKING
import logging
from dynamic_path_router import resolve_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from unified_event_bus import UnifiedEventBus
    from task_handoff_bot import WorkflowRecord

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly
    import networkx as nx  # type: ignore
    from networkx.readwrite import json_graph  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore
    json_graph = None  # type: ignore
    _HAS_NX = False


def estimate_impact_strength(from_id: str, to_id: str) -> tuple[float, str]:
    """Estimate how strongly ``from_id`` impacts ``to_id``.

    The score blends structural heuristics with live telemetry:

    * **Output coupling** – whether the outputs or queues of ``from_id`` feed
      into ``to_id``.
    * **API/service usage** – overlap in action chains or workflow steps.
    * **Resource contention** – competing for the same bots or other shared
      resources.
    * **Runtime signals** – historical ROI correlation from
      :mod:`roi_tracker` and queue load overlap via
      :mod:`resource_allocation_bot` when available.

    The function returns a normalised weight in ``[0, 1]`` along with a label
    describing which heuristic contributed the most.  Missing runtime metrics
    are simply ignored so the calculation falls back to the structural
    heuristics rather than failing.  When the required workflow information is
    unavailable a default ``(1.0, "unknown")`` is returned.
    """

    try:  # Local imports are used to avoid heavy start-up costs when optional
        from task_handoff_bot import WorkflowDB  # type: ignore
    except Exception:  # pragma: no cover - WorkflowDB unavailable
        return 1.0, "unknown"

    try:
        db = WorkflowDB()
    except Exception:  # pragma: no cover - database could not be opened
        return 1.0, "unknown"

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
        return 1.0, "unknown"

    def _jaccard(seq1: Any, seq2: Any) -> float:
        s1, s2 = set(seq1 or []), set(seq2 or [])
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0

    # Resource contention via shared bots/queues or other resources
    resource_contention = _jaccard(a.assigned_bots, b.assigned_bots)

    # API/service overlap from action chains or workflow steps
    api_coupling = _jaccard(
        a.action_chains or a.workflow, b.action_chains or b.workflow
    )

    # Optional vector similarity using workflow_vectorizer for a softer signal
    try:  # pragma: no cover - optional dependency
        from workflow_vectorizer import WorkflowVectorizer  # type: ignore

        vec = WorkflowVectorizer().fit([a.__dict__, b.__dict__])
        v1 = vec.transform(a.__dict__)
        v2 = vec.transform(b.__dict__)
        dot = sum(x * y for x, y in zip(v1, v2))
        norm = math.sqrt(sum(x * x for x in v1)) * math.sqrt(sum(y * y for y in v2))
        if norm:
            api_coupling = max(api_coupling, dot / norm)
    except Exception:
        pass

    # Output coupling – check if the last step of ``a`` feeds ``b``
    output_coupling = 0.0
    try:
        last_step = (a.workflow or a.task_sequence or [None])[-1]
        if last_step:
            if last_step in (b.argument_strings or []):
                output_coupling = 1.0
            elif (
                b.workflow or b.task_sequence or []
            ) and last_step == (b.workflow or b.task_sequence)[0]:
                output_coupling = 0.5
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Runtime data: ROI history correlation
    roi_corr: Optional[float] = None
    try:  # pragma: no cover - optional runtime telemetry
        from roi_tracker import ROITracker  # type: ignore

        tracker = ROITracker()
        try:
            tracker.load_history(str(resolve_path("sandbox_data/roi_history.json")))
        except Exception:
            pass
        hist_a = tracker.final_roi_history.get(str(from_id), [])
        hist_b = tracker.final_roi_history.get(str(to_id), [])
        if len(hist_a) > 1 and len(hist_b) > 1:
            mean_a = sum(hist_a) / len(hist_a)
            mean_b = sum(hist_b) / len(hist_b)
            num = sum(
                (x - mean_a) * (y - mean_b)
                for x, y in zip(hist_a[-50:], hist_b[-50:])
            )
            den = math.sqrt(sum((x - mean_a) ** 2 for x in hist_a[-50:])) * math.sqrt(
                sum((y - mean_b) ** 2 for y in hist_b[-50:])
            )
            if den:
                roi_corr = abs(num / den)
    except Exception:
        roi_corr = None

    # Runtime data: queue load overlap via allocation history
    queue_overlap: Optional[float] = None
    try:  # pragma: no cover - optional runtime telemetry
        from resource_allocation_bot import AllocationDB  # type: ignore

        adb = AllocationDB()

        def _avg_active(wid: str) -> float:
            try:
                row = adb.conn.execute(
                    "SELECT AVG(active) FROM allocations WHERE bot=?", (wid,)
                ).fetchone()
                return float(row[0] or 0.0)
            except Exception:
                return 0.0

        load_a = _avg_active(str(from_id))
        load_b = _avg_active(str(to_id))
        if load_a or load_b:
            queue_overlap = min(load_a, load_b)
    except Exception:
        queue_overlap = None

    # Aggregate scores, ignoring unavailable metrics
    scores: Dict[str, float] = {
        "resource": resource_contention,
        "api": api_coupling,
        "output": output_coupling,
    }
    if roi_corr is not None:
        scores["roi_corr"] = roi_corr
    if queue_overlap is not None:
        scores["queue"] = queue_overlap

    weight = sum(scores.values()) / len(scores) if scores else 1.0
    dep_type = max(scores, key=scores.get) if any(scores.values()) else "none"
    return max(0.0, min(1.0, weight)), dep_type


def estimate_edge_weight(from_id: str, to_id: str) -> float:
    """Backward compatible wrapper returning only the weight."""

    weight, _dep = estimate_impact_strength(from_id, to_id)
    return weight


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
        ``sandbox_data/workflow_graph.json`` is used.
    """

    def __init__(self, path: Optional[str] = None, *, db_path: Optional[str] = None) -> None:
        self.path = path or resolve_path("sandbox_data/workflow_graph.json")
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
                self.save()

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
            self.save()

            # Propagate the change through the dependency graph so
            # downstream modules can react to the updated workflow.
            try:
                roi_delta = float(event.get("roi_delta", 0.0) or 0.0)
                synergy_delta = float(event.get("synergy_delta", 0.0) or 0.0)
                impacts = self.simulate_impact_wave(wid, roi_delta, synergy_delta)
                logger.info("Impact wave from %s: %s", wid, impacts)
                bus.publish(
                    "workflows:impact_wave",
                    {"start_id": wid, "impact_map": impacts},
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("failed to simulate impact wave for %s: %s", wid, exc)

        def _on_remove(_topic: str, event: object) -> None:
            wid = _get_id(event)
            if wid is None:
                return
            self.remove_workflow(wid)
            self.refresh_edges()
            self.save()

        bus.subscribe("workflows:new", _on_new)
        bus.subscribe("workflows:update", _on_update)
        bus.subscribe("workflows:deleted", _on_remove)
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

    def update(self, workflow_id: str, change_type: str) -> None:
        """Apply a workflow change and persist the graph.

        Parameters
        ----------
        workflow_id:
            Identifier of the workflow being modified.
        change_type:
            Nature of the change – ``"add"``, ``"update"``, ``"remove"`` or
            ``"refactor"``.  Synonymous variations like ``"new"`` or
            ``"deleted"`` are also accepted.
        """

        ctype = change_type.lower()
        if ctype in {"add", "new"}:
            self.add_workflow(workflow_id)
            try:
                self.update_dependencies(workflow_id)
            except Exception:
                pass
        elif ctype in {"update", "updated"}:
            try:
                self.update_dependencies(workflow_id)
            except Exception:
                pass
        elif ctype in {"remove", "deleted", "delete", "refactor"}:
            self.remove_workflow(workflow_id)
            try:
                self.refresh_edges()
            except Exception:
                pass
        self.save()

    # ------------------------------------------------------------------
    # Automatic population from WorkflowDB
    # ------------------------------------------------------------------
    @staticmethod
    def _derive_edge_weights(
        a: "WorkflowRecord", b: "WorkflowRecord"
    ) -> tuple[float, float, float]:
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
                try:
                    self.add_dependency(
                        src_id,
                        dst_id,
                        dependency_type="derived",
                        resource_weight=res,
                        data_weight=data,
                        logical_weight=logical,
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
        dependency_type: Optional[str] = None,
        resource_weight: Optional[float] = None,
        data_weight: Optional[float] = None,
        logical_weight: Optional[float] = None,
        **attrs: Any,
    ) -> None:
        """Add a dependency edge between workflows.

        Parameters
        ----------
        src, dst:
            Source and destination workflow ids.
        impact_weight:
            Optional explicit aggregate edge weight. When omitted the value is
            derived from the individual ``*_weight`` parameters or, if those are
            not provided, heuristics from :func:`estimate_impact_strength` are
            used.
        dependency_type:
            Label describing the dependency category. When omitted the value
            returned from :func:`estimate_impact_strength` is stored.
        resource_weight, data_weight, logical_weight:
            Optional per-type weights capturing resource contention, data flow
            and logical dependency strengths respectively. When ``impact_weight``
            is not supplied these are averaged to form the overall edge weight.
        attrs:
            Additional edge attributes to persist on the graph.
        """
        weights = [
            w for w in (resource_weight, data_weight, logical_weight) if w is not None
        ]
        if impact_weight is None:
            if weights:
                impact_weight = sum(weights) / len(weights)
            else:
                impact_weight = estimate_edge_weight(src, dst)
            if dependency_type is None:
                _, dependency_type = estimate_impact_strength(src, dst)
        else:
            # Mark edges with manual weights so refresh routines skip them
            attrs.setdefault("manual_weight", True)
            if dependency_type is None:
                dependency_type = "manual"
        data: Dict[str, Any] = {
            "impact_weight": impact_weight,
            "dependency_type": dependency_type,
            "resource_weight": resource_weight or 0.0,
            "data_weight": data_weight or 0.0,
            "logical_weight": logical_weight or 0.0,
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
        # Refresh dependencies based on new workflow state and propagate the
        # refresh to downstream nodes so their edge weights remain consistent.
        try:
            wid = str(workflow_id)
            self.update_dependencies(wid)

            visited: set[str] = {wid}
            q: deque[str] = deque([wid])
            while q:
                current = q.popleft()
                with self._lock:
                    if _HAS_NX:
                        downstream = list(self.graph.successors(current))
                    else:
                        downstream = list(
                            self.graph.get("edges", {}).get(current, {}).keys()
                        )
                for nxt in downstream:
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    try:
                        self.update_dependencies(nxt)
                    except Exception:
                        continue
                    q.append(nxt)
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
                w, dep = estimate_impact_strength(workflow_id, node)
                if w > 0:
                    if _HAS_NX:
                        self.graph.add_edge(
                            workflow_id,
                            node,
                            impact_weight=w,
                            dependency_type=dep,
                        )
                        if self._graph_has_cycle():
                            self.graph.remove_edge(workflow_id, node)
                    else:
                        edges.setdefault(workflow_id, {})[node] = {
                            "impact_weight": w,
                            "dependency_type": dep,
                        }
                        if self._graph_has_cycle():
                            edges[workflow_id].pop(node, None)
                w, dep = estimate_impact_strength(node, workflow_id)
                if w > 0:
                    if _HAS_NX:
                        self.graph.add_edge(
                            node,
                            workflow_id,
                            impact_weight=w,
                            dependency_type=dep,
                        )
                        if self._graph_has_cycle():
                            self.graph.remove_edge(node, workflow_id)
                    else:
                        edges.setdefault(node, {})[workflow_id] = {
                            "impact_weight": w,
                            "dependency_type": dep,
                        }
                        if self._graph_has_cycle():
                            edges[node].pop(workflow_id, None)
            self._save_unlocked()

    def simulate_impact_wave(
        self,
        start_id: str,
        roi_delta: float,
        synergy_delta: float,
        *,
        resource_damping: float = 1.0,
        data_damping: float = 1.0,
        logical_damping: float = 1.0,
    ) -> Dict[str, Dict[str, float]]:
        """Propagate ROI and synergy deltas through the graph.

        Parameters
        ----------
        start_id:
            Workflow identifier where the change originates.
        roi_delta, synergy_delta:
            Initial metric deltas for ``start_id``.
        resource_damping, data_damping, logical_damping:
            Damping multipliers applied to ``resource_weight``, ``data_weight`` and
            ``logical_weight`` respectively when combining edge weights.  Values
            below ``1`` attenuate the influence of the corresponding dependency
            type while values above ``1`` amplify it.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping of workflow ids to projected ``roi`` and ``synergy`` deltas.
        """

        start_id = str(start_id)

        def _edge_weight(data: Dict[str, Any]) -> float:
            rw = float(data.get("resource_weight", 0.0) or 0.0)
            dw = float(data.get("data_weight", 0.0) or 0.0)
            lw = float(data.get("logical_weight", 0.0) or 0.0)
            num = 0.0
            denom = 0
            if rw > 0:
                num += rw * resource_damping
                denom += 1
            if dw > 0:
                num += dw * data_damping
                denom += 1
            if lw > 0:
                num += lw * logical_damping
                denom += 1
            if denom == 0:
                return float(data.get("impact_weight", 0.0) or 0.0)
            return num / denom

        def _outgoing(src: str):
            """Yield ``(dst, weight)`` pairs for positive weighted edges."""

            if _HAS_NX:
                for _, dst, data in self.graph.out_edges(src, data=True):
                    weight = _edge_weight(data)
                    if weight > 0.0:
                        yield dst, weight
            else:  # pragma: no cover - simple dict based fallback
                edges = self.graph.get("edges", {})
                for dst, data in edges.get(src, {}).items():
                    weight = _edge_weight(data)
                    if weight > 0.0:
                        yield dst, weight

        # Determine all nodes reachable from ``start_id`` following weighted
        # edges.  This restricts the subgraph considered in the simulation and
        # keeps the topological sort small.
        reachable: set[str] = {start_id}
        stack = [start_id]
        while stack:
            node = stack.pop()
            for nbr, _ in _outgoing(node):
                if nbr not in reachable:
                    reachable.add(nbr)
                    stack.append(nbr)

        # Topologically order the subgraph induced by the reachable nodes so we
        # always process predecessors before their dependants.
        if _HAS_NX:
            sub = self.graph.subgraph(reachable).copy()
            order = list(nx.topological_sort(sub))
        else:  # pragma: no cover - manual topo sort for fallback graph format
            edges = self.graph.get("edges", {})
            indeg: Dict[str, int] = {n: 0 for n in reachable}
            for src in reachable:
                for dst, _data in edges.get(src, {}).items():
                    if dst in indeg:
                        indeg[dst] += 1
            q: deque[str] = deque([n for n, d in indeg.items() if d == 0])
            order: list[str] = []
            while q:
                n = q.popleft()
                order.append(n)
                for dst, _data in edges.get(n, {}).items():
                    if dst in indeg:
                        indeg[dst] -= 1
                        if indeg[dst] == 0:
                            q.append(dst)

        impacts: Dict[str, Dict[str, float]] = {
            start_id: {"roi": roi_delta, "synergy": synergy_delta}
        }

        for node in order:
            current = impacts.get(node, {"roi": 0.0, "synergy": 0.0})
            for dst, weight in _outgoing(node):
                if dst not in reachable:
                    # Only propagate to nodes in the connected subgraph.
                    continue
                dest = impacts.setdefault(dst, {"roi": 0.0, "synergy": 0.0})
                dest["roi"] += current["roi"] * weight
                dest["synergy"] += current["synergy"] * weight

        logger.debug("Impact wave from %s: %s", start_id, impacts)
        return impacts

    def refresh_edges(self) -> None:
        """Recalculate impact weights for all dependency edges."""
        with self._lock:
            if _HAS_NX:
                for src, dst, data in list(self.graph.edges(data=True)):
                    if data.get("manual_weight"):
                        continue
                    w, dep = estimate_impact_strength(src, dst)
                    self.graph[src][dst]["impact_weight"] = w
                    self.graph[src][dst]["dependency_type"] = dep
            else:
                edges: Dict[str, Dict[str, Dict[str, Any]]] = self.graph.get("edges", {})
                for src, targets in edges.items():
                    for dst, data in list(targets.items()):
                        if data.get("manual_weight"):
                            continue
                        w, dep = estimate_impact_strength(src, dst)
                        data["impact_weight"] = w
                        data["dependency_type"] = dep
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
            if _HAS_NX and json_graph:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    return json_graph.node_link_graph(data, directed=True, edges="edges")
                except Exception:
                    pass
            else:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        return json.load(fh)
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
        if _HAS_NX and json_graph:
            data = json_graph.node_link_data(self.graph, edges="edges")
        else:
            data = self.graph
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        base, _ = os.path.splitext(self.path)
        snap_path = f"{base}.{ts}.json"
        try:
            with open(snap_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception:
            pass


def build_graph(db_path: Optional[str] = None, path: Optional[str] = None):
    """Rebuild the workflow dependency graph from the database.

    Parameters
    ----------
    db_path:
        Optional path to the workflows database. When omitted the default
        location used by :class:`task_handoff_bot.WorkflowDB` is consulted.
    path:
        Optional override for where the resulting graph should be persisted.

    Returns
    -------
    networkx.DiGraph
        The freshly constructed dependency graph.
    """

    wg = WorkflowGraph(path=path)
    if _HAS_NX:
        wg.graph = nx.DiGraph()
    else:  # pragma: no cover - fallback when networkx missing
        wg.graph = {"nodes": {}, "edges": {}}
    wg.populate_from_db(db_path)
    wg.save()
    return wg.graph


def load_graph(path: Optional[str] = None):
    """Load a previously persisted workflow dependency graph."""

    wg = WorkflowGraph(path=path, db_path=None)
    return wg.graph


__all__ = [
    "WorkflowGraph",
    "estimate_edge_weight",
    "estimate_impact_strength",
    "build_graph",
    "load_graph",
]
