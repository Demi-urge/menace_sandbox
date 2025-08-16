from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class ReplayExchange:
    """Single exchange captured for experience replay."""

    text: str
    reward_stack: List[float]
    context: List[float]
    flags: Dict[str, bool]
    timestamp: float = field(default_factory=time.time)
    category: str = "generic"
    weight: float = 1.0


class ExperienceReplayBuffer:
    """Rolling buffer with priority sampling and diversity checks."""

    def __init__(self, max_size: int = 1000, decay: float = 0.98) -> None:
        self.max_size = max_size
        self.decay = decay
        self.buffer: Deque[ReplayExchange] = deque()
        self.penalties: Dict[str, float] = {}
        self._counter = 0

    # ------------------------------------------------------------------
    def add_exchange(
        self,
        text: str,
        reward_stack: List[float],
        context: List[float],
        *,
        flags: Optional[Dict[str, bool]] = None,
        category: str = "generic",
        weight: float = 1.0,
    ) -> int:
        """Add a new exchange to the buffer."""
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()
        rec = ReplayExchange(
            text=text,
            reward_stack=reward_stack[:],
            context=context[:],
            flags=flags or {},
            category=category,
            weight=weight,
        )
        self.buffer.append(rec)
        self._counter += 1
        if rec.flags.get("error"):
            self.penalties[category] = min(1.0, self.penalties.get(category, 0.0) + 0.1)
        return self._counter - 1

    # ------------------------------------------------------------------
    def _base_weight(self, rec: ReplayExchange) -> float:
        base = sum(rec.reward_stack) + 1.0
        if rec.flags.get("rage_quit") or rec.flags.get("viral") or rec.flags.get("confessional"):
            base *= 2.0
        if rec.flags.get("error"):
            base *= 1.5
        return base * rec.weight

    def _decay(self) -> None:
        for rec in self.buffer:
            if not (rec.flags.get("rage_quit") or rec.flags.get("viral") or rec.flags.get("confessional")):
                rec.weight *= self.decay

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> List[ReplayExchange]:
        """Sample exchanges using priority and diversity filters."""
        if not self.buffer:
            return []
        self._decay()
        experiences = list(self.buffer)
        weights = [self._base_weight(r) for r in experiences]
        selected: List[ReplayExchange] = []
        categories: Dict[str, int] = {}
        for _ in range(min(batch_size, len(experiences))):
            if not experiences:
                break
            idx = random.choices(range(len(experiences)), weights=weights, k=1)[0]
            rec = experiences.pop(idx)
            weights.pop(idx)
            if categories.get(rec.category, 0) > batch_size // 2 and experiences:
                continue
            categories[rec.category] = categories.get(rec.category, 0) + 1
            selected.append(rec)
        return selected

    # ------------------------------------------------------------------
    def retrofit_reward(self, index: int, new_reward: float) -> None:
        """Retroactively adjust the latest reward in the stack."""
        if 0 <= index < len(self.buffer):
            rec = list(self.buffer)[index]
            if rec.reward_stack:
                rec.reward_stack[-1] = new_reward
                rec.weight = self._base_weight(rec)

    # ------------------------------------------------------------------
    def confidence_scale(self, category: str) -> float:
        penalty = self.penalties.get(category, 0.0)
        return max(0.0, 1.0 - penalty)
