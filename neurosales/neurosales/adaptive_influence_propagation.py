from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class InfluenceEvent:
    node: str
    delta: float
    timestamp: float


class AdaptiveInfluencePropagator:
    """Propagate influence changes over time through a network."""

    def __init__(
        self,
        edges: Optional[Dict[str, Dict[str, float]]] = None,
        *,
        base_decay: float = 0.99,
        lag_factor: float = 0.8,
    ) -> None:
        self.edges = edges or {}
        self.base_decay = base_decay
        self.lag_factor = lag_factor
        self.influence: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = {}
        self.events: List[InfluenceEvent] = []

    # ------------------------------------------------------------------
    def _leverage(self, node: str) -> float:
        degree = len(self.edges.get(node, {}))
        return 1.0 + 0.1 * degree

    def _volatility(self, node: str) -> float:
        h = self.history.get(node, [])
        if len(h) < 2:
            return 1.0
        avg = sum(h) / len(h)
        var = sum((x - avg) ** 2 for x in h) / len(h)
        return 1.0 + min(1.0, math.sqrt(var))

    # ------------------------------------------------------------------
    def record_shift(self, node: str, delta: float) -> None:
        """Register a major shift for propagation."""
        self.influence[node] = self.influence.get(node, 0.0) + delta
        self.history.setdefault(node, []).append(delta)
        self.events.append(InfluenceEvent(node, delta, time.time()))

    # ------------------------------------------------------------------
    def _apply(self, node: str, delta: float) -> None:
        self.influence[node] = self.influence.get(node, 0.0) + delta
        self.history.setdefault(node, []).append(delta)
        leverage = self._leverage(node)
        vol = self._volatility(node)
        for nbr, weight in self.edges.get(node, {}).items():
            ripple = delta * weight * leverage * vol
            self.influence[nbr] = self.influence.get(nbr, 0.0) + ripple
            self.history.setdefault(nbr, []).append(ripple)
            for nbr2, w2 in self.edges.get(nbr, {}).items():
                second = ripple * w2 * 0.5
                self.influence[nbr2] = self.influence.get(nbr2, 0.0) + second
                self.history.setdefault(nbr2, []).append(second)

    # ------------------------------------------------------------------
    def propagate(self) -> None:
        """Propagate all pending influence events with time decay."""
        now = time.time()
        for evt in list(self.events):
            elapsed = now - evt.timestamp
            factor = self.base_decay ** (elapsed / 60.0)
            delta = evt.delta * factor
            self._apply(evt.node, delta)
            evt.delta *= self.lag_factor
            evt.timestamp = now
            if abs(evt.delta) < 0.01:
                self.events.remove(evt)


__all__ = ["AdaptiveInfluencePropagator", "InfluenceEvent"]
