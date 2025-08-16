from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LineStats:
    """Track success/failure counts for a line."""

    success: float = 1.0
    fail: float = 1.0
    last_used: float = field(default_factory=time.time)


class AdaptiveExplorer:
    """Balance exploration and exploitation using Thompson Sampling."""

    def __init__(self, base_quota: float = 0.3) -> None:
        self.base_quota = base_quota
        self.topic_quota: Dict[str, float] = {}
        self.topic_confidence: Dict[str, float] = {}
        self.topic_engagement: Dict[str, float] = {}
        self.graph: Dict[str, Dict[str, LineStats]] = defaultdict(dict)
        self.regret: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.archetype_mod = {"newcomer": 1.2, "veteran": 0.8}

    # ------------------------------------------------------------------
    def _entropy(self, topic: str) -> float:
        stats = self.graph.get(topic, {})
        total = sum(ls.success + ls.fail for ls in stats.values()) or 1.0
        probs = [(ls.success + ls.fail) / total for ls in stats.values()]
        return -sum(p * math.log(p) for p in probs if p > 0.0)

    # ------------------------------------------------------------------
    def choose_line(
        self,
        user_id: str,
        archetype: str,
        topic: str,
        lines: List[str],
        is_fresh: List[bool],
        *,
        confidence: float = 0.5,
        engagement: float = 0.5,
    ) -> str:
        """Select a line balancing exploration and exploitation."""
        self.topic_confidence[topic] = confidence
        quota = self.topic_quota.get(topic, self.base_quota)
        quota *= self.archetype_mod.get(archetype, 1.0)
        if confidence > 0.7:
            quota *= 0.5
        if engagement < 0.4:
            quota = min(1.0, quota + 0.2)
        self.topic_quota[topic] = quota
        self.topic_engagement[topic] = engagement

        stats = self.graph[topic]
        scores: List[float] = []
        for line, fresh in zip(lines, is_fresh):
            st = stats.setdefault(line, LineStats())
            draw = random.betavariate(st.success, st.fail)
            score = draw + (quota if fresh else 0.0)
            scores.append(score)

        best_idx = max(range(len(lines)), key=lambda i: scores[i])
        chosen = lines[best_idx]

        for i, line in enumerate(lines):
            if i != best_idx and scores[i] > scores[best_idx]:
                self.regret[topic][line] = self.regret[topic].get(line, 0.0) + (
                    scores[i] - scores[best_idx]
                )
                if self.regret[topic][line] > 1.0:
                    stats[line].fail = max(1.0, stats[line].fail - 0.5)
                    self.regret[topic][line] = 0.0
        return chosen

    # ------------------------------------------------------------------
    def record_feedback(
        self, topic: str, line: str, *, success: bool, engagement: float
    ) -> None:
        """Update stats after observing the outcome."""
        stats = self.graph.setdefault(topic, {}).setdefault(line, LineStats())
        if success:
            stats.success += 1.0
        else:
            stats.fail += 1.0
        stats.last_used = time.time()
        prev = self.topic_engagement.get(topic, engagement)
        self.topic_engagement[topic] = prev * 0.8 + engagement * 0.2
        if self.topic_engagement[topic] < 0.4:
            self.topic_quota[topic] = min(
                1.0, self.topic_quota.get(topic, self.base_quota) + 0.1
            )
        self.topic_confidence[topic] = self.topic_confidence.get(topic, 0.5) * (
            1.0 if success else 0.9
        )
