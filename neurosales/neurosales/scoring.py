from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import logging

from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)


@dataclass(order=True)
class ResponseEntry:
    score: float
    response: str = field(compare=False)
    triggers: Dict[str, float] = field(default_factory=dict, compare=False)
    metrics: Dict[str, float] = field(default_factory=dict, compare=False)


class ResponsePriorityQueue:
    """Max-heap priority queue adapting scores with reinforcement signals."""

    def __init__(self) -> None:
        self.heap: List[Tuple[float, ResponseEntry]] = []

    def _total_score(self, triggers: Dict[str, float], metrics: Dict[str, float]) -> float:
        base = sum(triggers.values())
        reinforcement = (
            metrics.get("ctr", 0.0)
            + metrics.get("bounce_reduction", 0.0)
            + metrics.get("engagement", 0.0)
        )
        return base + reinforcement

    def add_response(self, response: str, triggers: Dict[str, float]) -> None:
        score = self._total_score(triggers, {})
        entry = ResponseEntry(score=score, response=response, triggers=triggers)
        heapq.heappush(self.heap, (-score, entry))

    def update_metrics(
        self,
        response: str,
        *,
        ctr: float = 0.0,
        bounce_reduction: float = 0.0,
        engagement: float = 0.0,
    ) -> None:
        for i, (_, entry) in enumerate(self.heap):
            if entry.response == response:
                entry.metrics["ctr"] = entry.metrics.get("ctr", 0.0) + ctr
                entry.metrics["bounce_reduction"] = entry.metrics.get("bounce_reduction", 0.0) + bounce_reduction
                entry.metrics["engagement"] = entry.metrics.get("engagement", 0.0) + engagement
                new_score = self._total_score(entry.triggers, entry.metrics)
                self.heap[i] = (-new_score, entry)
                heapq.heapify(self.heap)
                break

    def pop_best(self) -> Optional[str]:
        if not self.heap:
            return None
        _, entry = heapq.heappop(self.heap)
        return entry.response


class CandidateResponseScorer:
    """Evaluate candidate responses along multiple feature axes."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = {
            "semantic": 0.3,
            "emotion": 0.2,
            "engagement": 0.2,
            "personal": 0.2,
            "novelty": 0.1,
        }
        if weights:
            self.weights.update(weights)

        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - optional heavy deps
            np = None  # type: ignore
        self._np = np

        try:
            from sklearn.linear_model import LinearRegression
        except Exception:  # pragma: no cover - optional heavy dep
            LinearRegression = None  # type: ignore
        self._lr = LinearRegression() if LinearRegression else None
        self._model_loaded = False
        self._X: List[List[float]] = []
        self._y: List[float] = []

        if self._lr is not None:
            try:
                from joblib import load  # type: ignore

                model_path = resolve_path("neurosales") / "engagement_model.joblib"
                if model_path.exists():
                    self._lr = load(model_path)
                    self._model_loaded = True
            except Exception:  # pragma: no cover - fallback to heuristic
                self._model_loaded = False

        try:  # pragma: no cover - optional dependency
            from .sentiment import SentimentAnalyzer
        except Exception:  # pragma: no cover - fallback when sentiment module missing
            class SentimentAnalyzer:  # type: ignore[misc]
                def analyse(self, text: str):  # noqa: D401 - simple stub
                    return 0.0, []

        try:  # pragma: no cover - optional dependency
            from .user_preferences import PreferenceProfile
        except Exception:  # pragma: no cover - fallback when preferences missing
            class PreferenceProfile:  # type: ignore[misc]
                embedding: List[float] = []
                archetype: str = ""

        self.sentiment = SentimentAnalyzer()
        self.PreferenceProfile = PreferenceProfile

        self.policy_params = None
        try:
            weights_path = resolve_path("neurosales") / "policy_params.json"
            if weights_path.exists():
                with open(weights_path) as pf:
                    self.policy_params = json.load(pf)
        except Exception:
            self.policy_params = None

    # ------------------------------------------------------------------
    def _vectorize(self, text: str) -> List[float]:
        from .embedding import embed_text

        return embed_text(text)

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        if self._np is not None:
            A = self._np.array(a)
            B = self._np.array(b)
            denom = float(self._np.linalg.norm(A) * self._np.linalg.norm(B))
            if denom == 0.0:
                return 0.0
            return float(self._np.dot(A, B) / denom)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        denom = norm_a * norm_b
        return dot / denom if denom else 0.0

    def _predict_engagement(self, features: List[float]) -> float:
        if self._lr and self._np is not None and self._model_loaded:
            import numpy as np

            arr = np.array([features], dtype=float)
            try:
                return float(self._lr.predict(arr)[0])
            except Exception as e:  # pragma: no cover - prediction failure
                logger.exception("Engagement prediction failed")
                raise RuntimeError("Engagement prediction failed") from e
        # simple heuristic when no trained model is available
        return sum(features) / (len(features) or 1)

    def log_engagement(self, features: List[float], engagement: float) -> None:
        if self._lr and self._np is not None:
            self._X.append(features)
            self._y.append(engagement)
            if len(self._X) >= 5:
                import numpy as np

                X = np.array(self._X)
                y = np.array(self._y)
                self._lr.fit(X, y)

    def _novelty(self, text: str, history: List[str]) -> float:
        tokens = set(text.lower().split())
        if not history:
            return 1.0
        overlaps = []
        for h in history:
            ht = set(h.lower().split())
            union = tokens | ht
            inter = tokens & ht
            overlaps.append(len(inter) / (len(union) or 1))
        avg = sum(overlaps) / len(overlaps)
        return 1.0 - avg

    # ------------------------------------------------------------------
    def score_candidates(
        self,
        user_message: str,
        candidates: List[str],
        user_profile: "PreferenceProfile",
        history: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        history = history or []
        msg_emb = self._vectorize(user_message)
        user_sent, _ = self.sentiment.analyse(user_message)
        scores: Dict[str, float] = {}
        for cand in candidates:
            emb = self._vectorize(cand)
            semantic = self._cosine(msg_emb, emb)
            cand_sent, _ = self.sentiment.analyse(cand)
            emotional = 1.0 - abs(cand_sent - user_sent)
            engagement_features = [len(cand), cand.count("!"), cand.count("?")]
            engagement = self._predict_engagement(engagement_features)
            personal = self._cosine(user_profile.embedding, emb)
            novelty = self._novelty(cand, history)
            policy_adj = 0.0
            if self.policy_params:
                weights = self.policy_params[0]
                feats = [len(cand), cand.count("!"), cand.count("?")][: len(weights)]
                policy_adj = sum(w * f for w, f in zip(weights, feats)) / (len(weights) or 1)
            composite = (
                semantic * self.weights["semantic"]
                + emotional * self.weights["emotion"]
                + engagement * self.weights["engagement"]
                + personal * self.weights["personal"]
                + novelty * self.weights["novelty"]
                + policy_adj
            )
            scores[cand] = composite
        if not scores:
            return {}
        vals = list(scores.values())
        mn = min(vals)
        mx = max(vals)
        for k, v in scores.items():
            if mx != mn:
                scores[k] = (v - mn) / (mx - mn)
            else:
                scores[k] = 0.5
        return scores
