from __future__ import annotations

"""Suggest workflow chains based on embedding similarity and ROI metrics."""

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Tuple, Iterable, Set
import random
import json
from pathlib import Path
import sys

from dynamic_path_router import resolve_path
from vector_utils import cosine_similarity
from roi_results_db import ROIResultsDB
from workflow_stability_db import WorkflowStabilityDB
from logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from workflow_synergy_comparator import WorkflowSynergyComparator
except Exception:  # pragma: no cover - provide deterministic fallback
    logger.warning(
        "workflow_synergy_comparator import failed; entropy metrics disabled"
    )

    class WorkflowSynergyComparator:  # type: ignore[no-redef]
        @staticmethod
        def _entropy(*_, **__):
            return 0.0

try:  # pragma: no cover - optional dependency
    from task_handoff_bot import WorkflowDB  # type: ignore
except Exception:  # pragma: no cover - allow using a dummy DB in tests
    logger.warning("task_handoff_bot.WorkflowDB import failed; workflow DB disabled")
    WorkflowDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from code_database import CodeDB  # type: ignore
except Exception:  # pragma: no cover - used in tests without the real DB
    logger.warning("code_database import failed; context tag support disabled")
    CodeDB = None  # type: ignore
    sys.modules.pop("code_database", None)

try:  # pragma: no cover - compute default embedding path
    _CHAIN_EMBEDDINGS_PATH = resolve_path("sandbox_data/embeddings.jsonl")
except FileNotFoundError:  # pragma: no cover - file may not exist yet
    _CHAIN_EMBEDDINGS_PATH = resolve_path("sandbox_data") / "embeddings.jsonl"


def _load_chain_embeddings(
    path: Path = _CHAIN_EMBEDDINGS_PATH,
) -> List[Dict[str, Any]]:
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
    code_db: CodeDB | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple wiring
        if self.wf_db is None and WorkflowDB is not None:
            self.wf_db = WorkflowDB()
        if self.roi_db is None:
            self.roi_db = ROIResultsDB()
        if self.stability_db is None:
            self.stability_db = WorkflowStabilityDB()
        if self.code_db is None and CodeDB is not None:
            try:
                self.code_db = CodeDB()
            except Exception:
                self.code_db = None

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

    def _context_tags(self, workflow_id: str) -> List[str]:
        """Return CodeDB context tags for ``workflow_id``."""

        if not self.code_db:
            return []
        try:
            if hasattr(self.code_db, "get_context_tags"):
                tags = self.code_db.get_context_tags(workflow_id) or []
            elif hasattr(self.code_db, "context_tags"):
                tags = self.code_db.context_tags(workflow_id) or []
            else:
                tags = []
            return [str(t) for t in tags]
        except Exception:
            return []

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
        except Exception as exc:
            logger.warning("ROI delta fetch failed for %s: %s", workflow_id, exc)
            return 0.0
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
        self,
        vectors: List[Tuple[str, Sequence[float], float]],
        k: int,
        iterations: int = 10,
    ) -> List[List[str]]:
        weighted = [([w * x for x in vec], wid, w) for wid, vec, w in vectors]
        centroids = [list(vec) for vec, _, _ in random.sample(weighted, k)]
        assign = [0] * len(weighted)
        for _ in range(iterations):
            changed = False
            for i, (vec, _wid, _w) in enumerate(weighted):
                best = min(
                    range(k),
                    key=lambda j: 1 - cosine_similarity(vec, centroids[j]),
                )
                if assign[i] != best:
                    assign[i] = best
                    changed = True
            for j in range(k):
                members = [weighted[i] for i in range(len(weighted)) if assign[i] == j]
                if members:
                    total_w = sum(m[2] for m in members)
                    dims = zip(*[m[0] for m in members])
                    centroids[j] = [sum(d) / total_w for d in dims]
            if not changed:
                break
        clusters: Dict[int, List[str]] = {i: [] for i in range(k)}
        for idx, (_vec, wid, _w) in enumerate(weighted):
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
        vecs: List[Tuple[str, Sequence[float], float]] = []
        tags: Dict[str, Set[str]] = {}
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
            weight = (1.0 + roi) * stability
            vecs.append((str(wid), vec, weight))
            tags[str(wid)] = set(self._context_tags(wid))
        if not vecs:
            return suggestions
        k = min(3, len(vecs))
        clusters = self._kmeans(vecs, k)
        seqs: List[Tuple[float, List[str]]] = []
        for cl in clusters:
            ordered = sorted(cl, key=lambda w: scores.get(w, 0.0), reverse=True)
            avg = sum(scores.get(w, 0.0) for w in cl) / len(cl)
            tag_union: Set[str] = set()
            for w in cl:
                tag_union.update(tags.get(w, set()))
            if len(tag_union) > 1:
                avg *= 1.1
            elif len(tag_union) == 1:
                avg *= 0.9
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

        combined = suggestions + final[:remaining]
        valid = [c for c in combined if self._entropy(c) <= 1.5]

        return valid[:top_k]


_default_suggester: WorkflowChainSuggester | None = None


def suggest_chains(target_embedding: Sequence[float], top_k: int = 3) -> List[List[str]]:
    """Return workflow ID chains for ``target_embedding``."""

    global _default_suggester
    if _default_suggester is None:
        _default_suggester = WorkflowChainSuggester()
    return _default_suggester.suggest_chains(target_embedding, top_k)


__all__ = ["WorkflowChainSuggester", "suggest_chains"]
