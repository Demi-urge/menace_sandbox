from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .reward_ledger import CoinBalance


@dataclass
class InteractionRecord:
    user_id: str
    engagement: float
    sentiment: float
    personalization: float
    correct: bool
    context: str
    profile: str
    timestamp: float


class RewardKiln:
    """Simple EMA kiln for a single reward signal."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.value = 0.0

    def melt(self, metric: float) -> float:
        self.value = (1 - self.alpha) * self.value + self.alpha * metric
        return self.value


class HierarchicalRewardLearner:
    """Context and profile aware reward aggregator."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.kilns = {
            "green": RewardKiln(alpha),  # engagement
            "violet": RewardKiln(alpha),  # sentiment
            "gold": RewardKiln(alpha),  # personalization
            "iron": RewardKiln(alpha),  # correctness
        }
        self.context_weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"green": 1.0, "violet": 1.0, "gold": 1.0, "iron": 1.0}
        )
        self.history: List[InteractionRecord] = []

    # ------------------------------------------------------------------
    def record_interaction(
        self,
        user_id: str,
        *,
        engagement: float,
        sentiment: float,
        personalization: float,
        correct: bool,
        context: str = "general",
        profile: str = "",
    ) -> CoinBalance:
        """Process a single interaction and return weighted coin balance."""
        weights = self.context_weights[context].copy()
        if context == "emotional":
            weights["violet"] *= 1.5
        elif context == "tech_support":
            weights["iron"] *= 1.5

        if profile == "explorer":
            weights["green"] *= 2.0
        elif profile == "analyst":
            weights["iron"] *= 2.0

        g = self.kilns["green"].melt(engagement) * weights["green"]
        v = self.kilns["violet"].melt(sentiment) * weights["violet"]
        p = self.kilns["gold"].melt(personalization) * weights["gold"]
        c_val = 1.0 if correct else -1.0
        i = self.kilns["iron"].melt(c_val) * weights["iron"]

        self.history.append(
            InteractionRecord(
                user_id,
                engagement,
                sentiment,
                personalization,
                correct,
                context,
                profile,
                time.time(),
            )
        )

        return CoinBalance(green=g, violet=v, gold=p, iron=i)

    # ------------------------------------------------------------------
    def nightly_audit(self, threshold: float = 0.7) -> None:
        """Adjust context weights if high engagement pairs with mistakes."""
        for rec in self.history:
            if not rec.correct and (rec.engagement > threshold or rec.sentiment > threshold):
                w = self.context_weights[rec.context]
                w["green"] *= 0.9
                w["violet"] *= 0.9
                w["iron"] += 0.1
        self.history.clear()

