from __future__ import annotations

"""Predict modules with elevated error probabilities using clustering."""

from typing import Dict, List, TYPE_CHECKING
import logging
import random
import uuid

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    KMeans = None  # type: ignore

from menace.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:  # pragma: no cover - typing only
    try:
        from .error_bot import ErrorDB
    except Exception:  # pragma: no cover - typing fallback
        from error_bot import ErrorDB  # type: ignore

from vector_service import Retriever, FallbackResult

try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback
    class ErrorResult(Exception):
        """Fallback ErrorResult when vector service lacks explicit class."""

        pass


class ErrorClusterPredictor:
    """Cluster module telemetry to identify high-risk modules."""

    def __init__(
        self, graph: KnowledgeGraph, db: ErrorDB, retriever: Retriever | None = None
    ) -> None:
        self.graph = graph
        self.db = db
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def _module_vectors(self) -> tuple[List[str], List[List[float]]]:
        """Return modules and feature vectors derived from telemetry."""
        self.graph.update_error_stats(self.db)
        g = self.graph.graph
        if g is None:
            return [], []
        modules = [n for n in g.nodes if str(n).startswith("module:")]
        error_types = sorted(
            {
                e
                for m in modules
                for e in g.predecessors(m)
                if str(e).startswith("error_type:")
            }
        )
        if not modules or not error_types:
            return [], []
        vectors: List[List[float]] = []
        for m in modules:
            vec = [float(g.get_edge_data(e, m, {}).get("weight", 0.0)) for e in error_types]
            vectors.append(vec)
        return modules, vectors

    # ------------------------------------------------------------------
    def predict_high_risk_modules(
        self, *, min_cluster_size: int = 2, top_n: int = 5
    ) -> List[str]:
        """Return modules with elevated error probabilities."""
        modules, vectors = self._module_vectors()
        if not modules:
            return []
        labels: List[int] = []
        if hdbscan and len(vectors) >= min_cluster_size:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = list(clusterer.fit_predict(vectors))
            except Exception:  # pragma: no cover - optional dependency
                labels = []
        if not labels:
            n_clusters = max(1, len(modules) // max(1, min_cluster_size))
            if KMeans:
                km = KMeans(n_clusters=n_clusters, n_init="auto")  # type: ignore[arg-type]
            else:
                km = _SimpleKMeans(n_clusters=n_clusters)
            km.fit(vectors)
            labels = list(km.predict(vectors))
        cluster_scores: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        module_scores: Dict[str, float] = {}
        for m, vec, cid in zip(modules, vectors, labels):
            total = float(sum(vec))
            module_scores[m] = total
            cluster_scores[cid] = cluster_scores.get(cid, 0.0) + total
            counts[cid] = counts.get(cid, 0) + 1
        for cid in list(cluster_scores):
            if counts[cid]:
                cluster_scores[cid] /= counts[cid]
        scored = [
            (m, cluster_scores.get(lbl, 0.0), module_scores[m])
            for m, lbl in zip(modules, labels)
        ]
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        results = [m.split("module:")[1] for m, _, _ in scored[:top_n] if module_scores[m] > 0]
        if self.retriever is not None:
            for mod in results:
                try:
                    session_id = uuid.uuid4().hex
                    res = self.retriever.search(mod, top_k=1, session_id=session_id)
                    if isinstance(res, (FallbackResult, ErrorResult)):
                        if isinstance(res, FallbackResult):
                            self.logger.debug(
                                "retriever returned fallback for %s: %s",
                                mod,
                                getattr(res, "reason", ""),
                            )
                        else:
                            self.logger.debug("retriever returned fallback for %s", mod)
                except Exception:
                    self.logger.debug("retriever lookup failed", exc_info=True)
        return results


class _SimpleKMeans:
    """Fallback k-means clustering if scikit-learn is unavailable."""

    def __init__(self, n_clusters: int = 8, iters: int = 10) -> None:
        self.n_clusters = n_clusters
        self.iters = iters
        self.centers: List[List[float]] | None = None

    def fit(self, X: List[List[float]]) -> None:
        if not X:
            self.centers = []
            return
        self.centers = random.sample(X, min(self.n_clusters, len(X)))
        for _ in range(self.iters):
            clusters = [[] for _ in range(len(self.centers))]
            for vec in X:
                idx = self._closest(vec)[0]
                clusters[idx].append(vec)
            for i, cluster in enumerate(clusters):
                if cluster:
                    self.centers[i] = [sum(vals) / len(vals) for vals in zip(*cluster)]

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._closest(vec)[0] for vec in X]

    def _closest(self, vec: List[float]) -> tuple[int, float]:
        import math

        best = 0
        best_dist = float("inf")
        for i, c in enumerate(self.centers or []):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, c)))
            if dist < best_dist:
                best = i
                best_dist = dist
        return best, best_dist


__all__ = ["ErrorClusterPredictor"]
