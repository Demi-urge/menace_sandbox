from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class FeedbackRecord:
    """Single feedback entry for a response."""

    response: str
    rating: float
    user_id: str
    certainty: float
    correction: Optional[str] = None


class HumanFeedbackManager:
    """Aggregate real-time human feedback with normalization and decay."""

    def __init__(self, *, decay_rate: float = 0.99) -> None:
        self.decay_rate = decay_rate
        self.user_weights: Dict[str, float] = {}
        self.response_scores: Dict[str, float] = {}
        self.fine_tune_queue: List[Tuple[str, str]] = []
        self._mean = 0.0
        self._m2 = 0.0
        self._count = 0

    # ---------------------- user management ----------------------
    def register_user(self, user_id: str, weight: float = 1.0) -> None:
        self.user_weights[user_id] = max(0.1, weight)

    def _update_stats(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def _normalize(self, value: float) -> float:
        if self._count < 2:
            return value
        variance = self._m2 / (self._count - 1)
        if variance <= 0:
            return value
        stdev = math.sqrt(variance)
        limit = self._mean + 3 * stdev
        lower = self._mean - 3 * stdev
        if value > limit:
            return limit
        if value < lower:
            return lower
        return value

    # ---------------------- feedback handling ----------------------
    def record_feedback(
        self,
        response: str,
        *,
        rating: float,
        user_id: str,
        certainty: float,
        correction: Optional[str] = None,
    ) -> None:
        weight = self.user_weights.get(user_id, 1.0)
        score = rating * weight * (1.0 + certainty)
        score = self._normalize(score)
        self._update_stats(score)
        prev = self.response_scores.get(response, 0.0) * self.decay_rate
        self.response_scores[response] = prev + score
        if correction:
            self.fine_tune_queue.append((response, correction))

    def decay_scores(self) -> None:
        for key in list(self.response_scores.keys()):
            self.response_scores[key] *= self.decay_rate

    def adjust_ranking(self, scores: Dict[str, float]) -> List[str]:
        adjusted = {
            resp: base + self.response_scores.get(resp, 0.0)
            for resp, base in scores.items()
        }
        return sorted(adjusted, key=lambda r: adjusted[r], reverse=True)

    def mini_fine_tune(self, batch_size: int) -> List[Tuple[str, str]]:
        batch = self.fine_tune_queue[:batch_size]
        self.fine_tune_queue = self.fine_tune_queue[batch_size:]
        return batch
