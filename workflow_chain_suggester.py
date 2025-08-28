from __future__ import annotations

"""Suggest workflow chains based on embedding similarity and ROI metrics."""

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Tuple, Iterable
import random
import json
from pathlib import Path

from vector_utils import cosine_similarity
from roi_results_db import ROIResultsDB
from workflow_stability_db import WorkflowStabilityDB
from workflow_synergy_comparator import WorkflowSynergyComparator

try:  # pragma: no cover - optional dependency
    from task_handoff_bot import WorkflowDB  # type: ignore
except Exception:  # pragma: no cover - allow using a dummy DB in tests
    WorkflowDB = None  # type: ignore


def _load_chain_embeddings(path: Path = Path("embeddings.jsonl")) -> List[Dict[str, Any]]:
    """Return stored workflow chain embeddings with metadata."""

    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "workflow_chain":
                continue
            vec = [float(x) for x in rec.get("vector", [])]
            meta = rec.get("metadata", {}) or {}
            records.append(
                {
                    "id": str(rec.get("id", "")),
                    "vector": vec,
                    "roi": float(meta.get("roi", 0.0)),
                    "entropy": float(meta.get("entropy", 0.0)),
                }
            )
    return records


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
            # Penalise unstable workflows by reducing the weight to at most 0.5
            return 0.5 / (1.0 + max(0.0, ema))
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Chain mutation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def swap_steps(chain: Sequence[str], i: int, j: int) -> List[str]:
        """Return ``chain`` with steps ``i`` and ``j`` swapped."""

        seq = list(chain)
        if 0 <= i < len(seq) and 0 <= j < len(seq):
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    @staticmethod
    def split_sequence(chain: Sequence[str], index: int) -> List[List[str]]:
        """Split ``chain`` at ``index`` into two sequences."""

        seq = list(chain)
        index = max(0, min(index, len(seq)))
        return [seq[:index], seq[index:]]

    @staticmethod
    def merge_partial_chains(chains: Iterable[Sequence[str]]) -> List[str]:
        """Merge ``chains`` into a single sequence removing duplicates."""

        merged: List[str] = []
        for ch in chains:
            for step in ch:
                if step not in merged:
                    merged.append(str(step))
        return merged

    # ------------------------------------------------------------------
    def _roi_delta(self, workflow_id: str) -> float:
        """Return the ROI improvement delta for ``workflow_id``."""

        if self.roi_db is None:
            return 0.0
        try:
            trends = self.roi_db.fetch_trends(str(workflow_id))
            if len(trends) >= 2:
                return float(trends[-1].get("roi_gain", 0.0)) - float(
                    trends[-2].get("roi_gain", 0.0)
                )
        except Exception:
            pass
        return 0.0

    def _entropy(self, chain: Sequence[str]) -> float:
        spec = {"steps": [{"module": m} for m in chain]}
        try:
            return WorkflowSynergyComparator._entropy(spec)
        except Exception:
            return 0.0

    def _should_split(self, chain: Sequence[str]) -> bool:
        deltas = [self._roi_delta(w) for w in chain]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        entropy = self._entropy(chain)
        return avg_delta < 0.0 and entropy > 1.0

    def _should_merge(self, chains: List[List[str]]) -> bool:
        deltas = [self._roi_delta(w) for ch in chains for w in ch]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        entropy = self._entropy(self.merge_partial_chains(chains))
        return avg_delta > 0.0 and entropy < 1.0

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
        chain_recs = _load_chain_embeddings()
        scored: List[Tuple[float, List[str]]] = []
        for rec in chain_recs:
            sim = cosine_similarity(target_embedding, rec.get("vector", []))
            score = sim * (1.0 + rec.get("roi", 0.0)) * max(
                0.0, 1.0 - rec.get("entropy", 0.0)
            )
            scored.append((score, rec.get("id", "").split("->")))
        scored.sort(key=lambda x: x[0], reverse=True)
        suggestions = [chain for _, chain in scored[:top_k]]
        remaining = top_k - len(suggestions)
        if remaining <= 0 or self.wf_db is None:
            return suggestions

        raw = self.wf_db.search_by_vector(target_embedding, remaining * 5)
        scores: Dict[str, float] = {}
        vecs: List[Tuple[str, Sequence[float]]] = []
        for wid, dist in raw:
            vec = self.wf_db.get_vector(wid)
            if not vec:
                continue
            sim = 1.0 / (1.0 + float(dist))
            roi = max(0.0, self._roi_score(wid))
            stability = self._stability_weight(wid)
            delta = max(0.0, self._roi_delta(wid))
            score = sim * (1.0 + roi) * stability * (1.0 + delta)
            scores[str(wid)] = score
            vecs.append((str(wid), vec))
        if not vecs:
            return suggestions
        k = min(3, len(vecs))
        clusters = self._kmeans(vecs, k)
        seqs: List[Tuple[float, List[str]]] = []
        for cl in clusters:
            ordered = sorted(cl, key=lambda w: scores.get(w, 0.0), reverse=True)
            avg = sum(scores.get(w, 0.0) for w in cl) / len(cl)
            seqs.append((avg, ordered))
        seqs.sort(key=lambda x: x[0], reverse=True)
        chosen = [s for _, s in seqs[:remaining]]

        final: List[List[str]] = []
        for seq in chosen:
            if self._should_split(seq):
                parts = [p for p in self.split_sequence(seq, len(seq) // 2) if p]
                final.extend(parts)
            else:
                final.append(seq)

        if len(final) > 1 and self._should_merge(final):
            final = [self.merge_partial_chains(final)]

        return suggestions + final[:remaining]


_default_suggester: WorkflowChainSuggester | None = None


def suggest_chains(target_embedding: Sequence[float], top_k: int = 3) -> List[List[str]]:
    """Return workflow ID chains for ``target_embedding``."""

    global _default_suggester
    if _default_suggester is None:
        _default_suggester = WorkflowChainSuggester()
    return _default_suggester.suggest_chains(target_embedding, top_k)


__all__ = ["WorkflowChainSuggester", "suggest_chains"]
