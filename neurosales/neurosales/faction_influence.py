from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Faction:
    """Data for a faction/coalition."""

    rating: float = 1500.0
    stability: float = 1.0
    neural_hits: Dict[str, int] = field(
        default_factory=lambda: {"vmPFC": 0, "amygdala": 0, "NAcc": 0}
    )


class FactionInfluenceEngine:
    """Track power shifts between factions using Elo-style ratings."""

    def __init__(self, *, k: float = 32.0, prop_factor: float = 0.1) -> None:
        self.k = k
        self.prop_factor = prop_factor
        self.factions: Dict[str, Faction] = {}
        self.allies: Dict[str, Dict[str, float]] = {}
        self.coalitions: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def add_faction(self, name: str, *, rating: float = 1500.0) -> None:
        if name not in self.factions:
            self.factions[name] = Faction(rating=rating)
            self.allies.setdefault(name, {})

    # ------------------------------------------------------------------
    def form_alliance(self, a: str, b: str, *, weight: float = 1.0) -> None:
        self.add_faction(a)
        self.add_faction(b)
        self.allies[a][b] = weight
        self.allies[b][a] = weight

    # ------------------------------------------------------------------
    def form_coalition(self, members: List[str], target: str) -> None:
        for m in members:
            self.coalitions[m] = target

    # ------------------------------------------------------------------
    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def _acceleration(self, name: str) -> float:
        faction = self.factions[name]
        total = sum(faction.neural_hits.values())
        return 1.0 + 0.1 * total

    # ------------------------------------------------------------------
    def _propagate(self, name: str, delta: float) -> None:
        for nbr, weight in self.allies.get(name, {}).items():
            self.factions[nbr].rating += delta * self.prop_factor * weight

    # ------------------------------------------------------------------
    def record_interaction(
        self,
        winner: str,
        loser: str,
        *,
        margin: float = 1.0,
        validation: float = 0.0,
        triggers: Optional[List[str]] = None,
    ) -> None:
        self.add_faction(winner)
        self.add_faction(loser)
        w = self.factions[winner]
        loser_faction = self.factions[loser]

        if triggers:
            for t in triggers:
                if t in w.neural_hits:
                    w.neural_hits[t] += 1

        exp = self._expected(w.rating, loser_faction.rating)
        change = self.k * margin * (1.0 - exp)

        if self.coalitions.get(winner) == loser:
            change *= 1.2

        change *= self._acceleration(winner)
        w.rating += change + validation * (self.k / 8)
        loser_faction.rating -= change * self._acceleration(loser)

        self._propagate(winner, change)
        self._propagate(loser, -change)

        self._apply_instability()

    # ------------------------------------------------------------------
    def _apply_instability(self) -> None:
        if not self.factions:
            return
        avg = sum(f.rating for f in self.factions.values()) / len(self.factions)
        high = avg * 1.2
        low = avg * 0.8
        for f in self.factions.values():
            if f.rating > high:
                f.rating -= (f.rating - high) * 0.05
            elif f.rating < low:
                f.rating += (low - f.rating) * 0.05


__all__ = ["FactionInfluenceEngine", "Faction"]
