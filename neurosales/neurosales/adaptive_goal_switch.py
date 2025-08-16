from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple


@dataclass
class GoalLog:
    timestamp: float
    old: str
    new: str
    reason: str


class AdaptiveGoalSwitcher:
    """Switch conversation goals based on running performance metrics."""

    def __init__(self, start_goal: str = "convert", *, window: int = 3, stickiness: int = 2) -> None:
        self.current_goal = start_goal
        self.window = window
        self.stickiness = stickiness
        self.metrics: Deque[Tuple[float, float, float]] = deque(maxlen=window)
        self._since_switch = 0
        self.logs: List[GoalLog] = []

    def update_metrics(self, persuasion: float, rapport: float, entertainment: float) -> None:
        """Record new metrics and update goal likelihood."""
        self.metrics.append((persuasion, rapport, entertainment))
        self._since_switch += 1
        if len(self.metrics) >= self.window:
            self._recompute()

    # ------------------------------------------------------------------
    def _averages(self) -> Tuple[float, float, float]:
        p = r = e = 0.0
        for pp, rr, ee in self.metrics:
            p += pp
            r += rr
            e += ee
        n = len(self.metrics) or 1
        return p / n, r / n, e / n

    def _recompute(self) -> None:
        p, r, e = self._averages()
        probs = {
            "convert": 0.6 * p + 0.2 * r + 0.2 * e,
            "comfort": 0.6 * r + 0.2 * p + 0.2 * e,
            "entertain": 0.6 * e + 0.2 * r + 0.2 * p,
        }
        new_goal = max(probs, key=probs.get)
        if new_goal != self.current_goal and self._since_switch >= self.stickiness:
            reason = ""
            if self.current_goal == "convert" and new_goal == "comfort" and p < 0.5 and r > 0.5:
                reason = "persuasion stalled but rapport strong"
            elif self.current_goal == "convert" and new_goal == "entertain" and p < 0.5 and e > 0.5:
                reason = "engagement outweighed persuasion"
            else:
                reason = "shift in metrics"
            self.logs.append(GoalLog(time.time(), self.current_goal, new_goal, reason))
            self.current_goal = new_goal
            self._since_switch = 0

    def explain_last(self) -> str:
        """Return a human-friendly explanation of the last goal switch."""
        if not self.logs:
            return "No goal shifts yet."
        log = self.logs[-1]
        return f"Switched from {log.old} to {log.new} because {log.reason}."
