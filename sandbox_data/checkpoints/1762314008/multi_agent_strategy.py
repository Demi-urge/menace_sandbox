from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .faction_influence import FactionInfluenceEngine


@dataclass
class DiplomaticEvent:
    """Record of negotiations and alliances."""

    timestamp: float
    actor: str
    target: str
    action: str
    detail: str = ""


class DiplomaticMemory:
    """Store diplomatic interactions for later reference."""

    def __init__(self) -> None:
        self.events: List[DiplomaticEvent] = []

    def record(self, actor: str, target: str, action: str, detail: str = "") -> None:
        self.events.append(DiplomaticEvent(time.time(), actor, target, action, detail))

    def history(self, actor: str, target: str) -> List[DiplomaticEvent]:
        return [e for e in self.events if e.actor == actor and e.target == target]


class MultiAgentStrategy:
    """Manage alliances, negotiations and betrayals between factions."""

    def __init__(self, members: Dict[str, List[str]], engine: FactionInfluenceEngine | None = None) -> None:
        self.engine = engine or FactionInfluenceEngine()
        self.members = members
        self.vote_weight: Dict[str, Dict[str, float]] = {
            fac: {m: 1.0 for m in ms} for fac, ms in members.items()
        }
        self.betrayals: Dict[Tuple[str, str], int] = {}
        self.posture: Dict[str, float] = {}
        self.memory = DiplomaticMemory()

    # ------------------------------------------------------------------
    def _betrayal_risk(self, a: str, b: str) -> float:
        return 0.1 * self.betrayals.get((b, a), 0)

    # ------------------------------------------------------------------
    def record_betrayal(self, betrayer: str, victim: str) -> None:
        key = (betrayer, victim)
        self.betrayals[key] = self.betrayals.get(key, 0) + 1
        self.memory.record(betrayer, victim, "betrayal")

    # ------------------------------------------------------------------
    def propose_alliance(self, proposer: str, partner: str) -> bool:
        self.engine.add_faction(proposer)
        self.engine.add_faction(partner)
        gain = (self.engine.factions[partner].rating - 1500.0) / 1000.0
        risk = self._betrayal_risk(proposer, partner)
        posture = self.posture.get(proposer, 1.0)
        members = self.members.get(proposer, [])
        votes_for = 0.0
        votes_against = 0.0
        bias = 0.0
        for m in members:
            weight = self.vote_weight[proposer].get(m, 1.0)
            score = gain * posture - risk + bias
            if score >= 0:
                votes_for += weight
            else:
                votes_against += weight
            bias = 0.1 * (votes_for - votes_against)
        detail = f"for={votes_for:.2f} against={votes_against:.2f}"
        self.memory.record(proposer, partner, "proposed_alliance", detail)
        if votes_for >= votes_against:
            self.engine.form_alliance(proposer, partner)
            self.memory.record(proposer, partner, "alliance_formed")
            return True
        return False

    # ------------------------------------------------------------------
    def negotiate(self, a: str, b: str, offer: str = "", concession: str = "") -> bool:
        risk = self._betrayal_risk(a, b)
        combined = self.engine.factions.get(a, FactionInfluenceEngine().factions.get(a, None))
        if combined is None:
            self.engine.add_faction(a)
            combined_rating = 1500.0
        else:
            combined_rating = combined.rating
        combined_rating += self.engine.factions.get(b, FactionInfluenceEngine().factions.get(b, None)).rating if b in self.engine.factions else 1500.0
        success = combined_rating / 3000.0 - risk > 0
        detail = f"offer={offer};concession={concession};success={success}"
        self.memory.record(a, b, "negotiation", detail)
        if success:
            self.engine.form_alliance(a, b)
            self.memory.record(a, b, "alliance_formed")
        return success


__all__ = ["MultiAgentStrategy", "DiplomaticMemory", "DiplomaticEvent"]
