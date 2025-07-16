from __future__ import annotations

"""Q-learning policy predicting ROI improvement from self-improvement cycles."""

from typing import Dict, Tuple, Optional
import pickle
import os


class SelfImprovementPolicy:
    """Tiny reinforcement learning helper for the self-improvement engine."""

    def __init__(self, alpha: float = 0.5, gamma: float = 0.9, path: Optional[str] = None) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.values: Dict[Tuple[int, ...], float] = {}
        self.path = path
        if self.path:
            self.load(self.path)

    # ------------------------------------------------------------------
    def update(
        self,
        state: Tuple[int, ...],
        reward: float,
        next_state: Tuple[int, ...] | None = None,
    ) -> float:
        """Update ``state`` with ``reward`` and optional ``next_state``."""
        q = self.values.get(state, 0.0)
        next_best = max(self.values.get(next_state, 0.0) if next_state else 0.0, 0.0)
        q += self.alpha * (reward + self.gamma * next_best - q)
        self.values[state] = q
        if self.path:
            self.save(self.path)
        return q

    def score(self, state: Tuple[int, ...]) -> float:
        """Return the learned value for ``state``."""
        return self.values.get(state, 0.0)

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp:
            return
        try:
            with open(fp, "wb") as fh:
                pickle.dump(self.values, fh)
        except Exception:
            raise

    def load(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp or not os.path.exists(fp):
            return
        try:
            with open(fp, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                self.values = {tuple(k) if not isinstance(k, tuple) else k: float(v) for k, v in data.items()}
        except Exception:
            raise


__all__ = ["SelfImprovementPolicy"]
