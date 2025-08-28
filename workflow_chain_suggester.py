from __future__ import annotations

"""Suggest workflow chains based on embedding similarity and ROI metrics."""

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Tuple
import random

from vector_utils import cosine_similarity
from roi_results_db import ROIResultsDB
from workflow_stability_db import WorkflowStabilityDB

try:  # pragma: no cover - optional dependency
    from task_handoff_bot import WorkflowDB  # type: ignore
except Exception:  # pragma: no cover - allow using a dummy DB in tests
    WorkflowDB = None  # type: ignore


@dataclass
class WorkflowChainSuggester:
    """Combine vector similarity with ROI and stability weights."""

    wf_db: Any | None = None
    roi_db: ROIResultsDB | None = None
    stability_db: WorkflowStabilityDB | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        if self.wf_db is None and WorkflowDB is not None:
            self.wf_db = WorkflowDB()
        if self.roi_db is None:
            self.roi_db = ROIResultsDB()
        if self.stability_db is None:
            self.stability_db = WorkflowStabilityDB()

    # ------------------------------------------------------------------
    def _roi_score(self, workflow_id: str) -> float:
        if self.roi_db is None:
            return 0.0
        try:
            trends = self.roi_db.fetch_trends(str(workflow_id))
            if not trends:
                return 0.0
            return float(trends[-1].get("roi_gain", 0.0))
        except Exception:
            return 0.0

    def _stability_weight(self, workflow_id: str) -> float:
        if self.stability_db is None:
            return 1.0
        try:
            if self.stability_db.is_stable(str(workflow_id)):
                return 1.0
            ema, _ = self.stability_db.get_ema(str(workflow_id))
            return 1.0 / (1.0 + max(0.0, ema))
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    def _kmeans(
        self, vectors: List[Tuple[str, Sequence[float]]], k: int, iterations: int = 10
    ) -> List[List[str]]:
        centroids = [list(vec) for _, vec in random.sample(vectors, k)]
        assign = [0] * len(vectors)
        for _ in range(iterations):
            changed = False
            for i, (_, vec) in enumerate(vectors):
                best = min(
                    range(k),
                    key=lambda j: 1 - cosine_similarity(vec, centroids[j]),
                )
                if assign[i] != best:
                    assign[i] = best
                    changed = True
            for j in range(k):
                members = [vectors[i][1] for i in range(len(vectors)) if assign[i] == j]
                if members:
                    dims = zip(*members)
                    centroids[j] = [sum(d) / len(members) for d in dims]
            if not changed:
                break
        clusters: Dict[int, List[str]] = {i: [] for i in range(k)}
        for idx, (wid, _vec) in enumerate(vectors):
            clusters[assign[idx]].append(wid)
        return list(clusters.values())

    # ------------------------------------------------------------------
    def suggest_chains(
        self, target_embedding: Sequence[float], top_k: int = 3
    ) -> List[List[str]]:
        if self.wf_db is None:
            return []
        raw = self.wf_db.search_by_vector(target_embedding, top_k * 5)
        scores: Dict[str, float] = {}
        vecs: List[Tuple[str, Sequence[float]]] = []
        for wid, dist in raw:
            vec = self.wf_db.get_vector(wid)
            if not vec:
                continue
            sim = 1.0 / (1.0 + float(dist))
            roi = max(0.0, self._roi_score(wid))
            stability = self._stability_weight(wid)
            score = sim * (1.0 + roi) * stability
            scores[str(wid)] = score
            vecs.append((str(wid), vec))
        if not vecs:
            return []
        k = min(3, len(vecs))
        clusters = self._kmeans(vecs, k)
        seqs: List[Tuple[float, List[str]]] = []
        for cl in clusters:
            ordered = sorted(cl, key=lambda w: scores.get(w, 0.0), reverse=True)
            avg = sum(scores.get(w, 0.0) for w in cl) / len(cl)
            seqs.append((avg, ordered))
        seqs.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in seqs[:top_k]]


_default_suggester: WorkflowChainSuggester | None = None


def suggest_chains(target_embedding: Sequence[float], top_k: int = 3) -> List[List[str]]:
    """Return workflow ID chains for ``target_embedding``."""

    global _default_suggester
    if _default_suggester is None:
        _default_suggester = WorkflowChainSuggester()
    return _default_suggester.suggest_chains(target_embedding, top_k)


__all__ = ["WorkflowChainSuggester", "suggest_chains"]
