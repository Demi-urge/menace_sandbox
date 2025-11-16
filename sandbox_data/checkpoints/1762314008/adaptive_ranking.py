from __future__ import annotations

import random
from typing import Dict, List, Optional

from .rlhf import RLHFPolicyManager


class AdaptiveRanker:
    """Re-rank candidate responses using feedback driven EMAs and bandits."""

    def __init__(self, *, alpha: float = 0.3, exploration: float = 0.1) -> None:
        self.alpha = alpha
        self.exploration = exploration
        self.user_weights: Dict[str, Dict[str, float]] = {}
        self.bandits: Dict[str, RLHFPolicyManager] = {}

    # ------------------------------------------------------------------
    def _bandit(self, user_id: str) -> RLHFPolicyManager:
        b = self.bandits.get(user_id)
        if b is None:
            b = RLHFPolicyManager(exploration_rate=self.exploration)
            self.bandits[user_id] = b
        return b

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
    def rerank(
        self,
        user_id: str,
        scores: Dict[str, float],
        *,
        history: Optional[List[str]] = None,
        archetype: str = "",
        top_n: int = 5,
    ) -> List[str]:
        history = history or []
        if not scores:
            return []
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_n = min(max(3, top_n), 5)
        top = [c for c, _ in sorted_all[:top_n]]
        rest = [c for c, _ in sorted_all[top_n:]]

        weights = self.user_weights.get(user_id, {})
        bandit = self._bandit(user_id)
        final_scores: Dict[str, float] = {}
        for cand in top:
            base = scores[cand]
            base += weights.get(cand, 0.0)
            base += bandit.weights.get(cand, 0.0)
            if archetype == "analytical":
                base += len(cand) / 100.0
            elif archetype == "novelty":
                base += self._novelty(cand, history) * 0.2
            final_scores[cand] = base
        ranked = sorted(top, key=lambda x: final_scores[x], reverse=True)

        if rest and random.random() < self.exploration:
            novel_cand = max(rest, key=lambda c: self._novelty(c, history))
            if novel_cand not in ranked:
                ranked.insert(1 if len(ranked) > 1 else 0, novel_cand)
        return ranked

    # ------------------------------------------------------------------
    def update_with_feedback(
        self,
        user_id: str,
        response: str,
        *,
        ignored: bool = False,
        corrected: bool = False,
        engaged: bool = False,
        praised: bool = False,
        session_delta: float = 0.0,
        sentiment_delta: float = 0.0,
        followups: int = 0,
    ) -> None:
        weights = self.user_weights.setdefault(user_id, {})
        val = weights.get(response, 0.0)
        reward = 0.0
        if praised:
            reward += 0.5
        if engaged:
            reward += 0.3
        if corrected:
            reward -= 0.2
        if ignored:
            reward -= 0.4
        weights[response] = (1 - self.alpha) * val + self.alpha * reward

        bandit = self._bandit(user_id)
        synth_reward = 0.5 * session_delta + 0.3 * sentiment_delta + 0.2 * followups
        bandit.record_result(response, ctr=synth_reward, sentiment=sentiment_delta, session=session_delta)
