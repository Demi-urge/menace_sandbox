from __future__ import annotations

"""Analyze error clusters using :class:`KnowledgeGraph`."""

from typing import Dict, List
import logging

try:  # pragma: no cover - optional dependency
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None  # type: ignore

from .knowledge_graph import KnowledgeGraph
from .error_bot import ErrorDB


class ErrorClusterAnalyzer:
    """Identify and persist high-frequency error clusters."""

    def __init__(self, graph: KnowledgeGraph, db: ErrorDB) -> None:
        self.graph = graph
        self.db = db
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def analyze(self, min_cluster_size: int = 2) -> Dict[int, List[str]]:
        """Cluster error types based on telemetry frequency."""
        self.graph.update_error_stats(self.db)
        g = self.graph.graph
        if g is None or nx is None:
            return {}

        error_nodes = [n for n in g.nodes if n.startswith("error_type:")]
        if not error_nodes:
            return {}

        modules = sorted(
            {m for e in error_nodes for m in g.neighbors(e) if str(m).startswith("module:")}
        )
        vectors: List[List[float]] = []
        for e in error_nodes:
            vec = [float(g.get_edge_data(e, m, {}).get("weight", 0.0)) for m in modules]
            vec.append(float(g.nodes[e].get("weight", 0.0)))
            vectors.append(vec)

        labels: List[int] = []
        if hdbscan and len(vectors) >= min_cluster_size:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = list(clusterer.fit_predict(vectors))
            except Exception:  # pragma: no cover - optional dependency
                labels = []
        if not labels:
            tmp = nx.Graph()
            for e in error_nodes:
                tmp.add_node(e)
            for i, e1 in enumerate(error_nodes):
                for e2 in error_nodes[i + 1 :]:
                    weight = sum(
                        min(
                            g.get_edge_data(e1, m, {}).get("weight", 0),
                            g.get_edge_data(e2, m, {}).get("weight", 0),
                        )
                        for m in modules
                    )
                    if weight > 0:
                        tmp.add_edge(e1, e2, weight=weight)
            comps = list(nx.connected_components(tmp))
            cluster_map = {n: idx for idx, comp in enumerate(comps) for n in comp}
        else:
            cluster_map = {n: int(lbl) for n, lbl in zip(error_nodes, labels) if lbl >= 0}

        stored = {n.split("error_type:")[1]: cid for n, cid in cluster_map.items()}
        if stored:
            self.db.set_error_clusters(stored)

        result: Dict[int, List[str]] = {}
        for n, cid in cluster_map.items():
            result.setdefault(cid, []).append(n.split("error_type:")[1])
        return result

    # ------------------------------------------------------------------
    def likely_future_errors(self, bot_id: str, top_n: int = 3) -> List[str]:
        """Return probable future error types for ``bot_id``."""
        seen = self.db.get_bot_error_types(bot_id)
        if not seen:
            return []
        clusters = self.db.get_error_clusters()
        cluster_ids = {clusters[e] for e in seen if e in clusters}
        if not cluster_ids:
            return []
        related = self.db.get_error_types_for_clusters(list(cluster_ids))
        suggestions = [e for e in related if e not in seen]
        return suggestions[:top_n]


__all__ = ["ErrorClusterAnalyzer"]
