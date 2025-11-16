from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .engagement_graph import pagerank


@dataclass
class ArchetypeStats:
    """Interaction and reputation data for an archetype."""

    wins: int = 0
    losses: int = 0
    citations: int = 0
    alignment: float = 0.0
    reputation: float = 0.0


class SocialPowerRanker:
    """Rank archetypes by social power using PageRank and reputation signals."""

    def __init__(self, *, decay: float = 0.95) -> None:
        self.decay = decay
        self.archetypes: Dict[str, ArchetypeStats] = {}
        self.citation_edges: Dict[str, Dict[str, float]] = {}
        self.user_factions: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def _stats(self, name: str) -> ArchetypeStats:
        stats = self.archetypes.get(name)
        if stats is None:
            stats = ArchetypeStats()
            self.archetypes[name] = stats
        return stats

    def record_interaction(
        self,
        winner: str,
        loser: str,
        *,
        cited_by: Optional[List[str]] = None,
        alignment: float = 1.0,
        contradiction: bool = False,
        behavioral_response: bool = True,
    ) -> None:
        """Record a conversational exchange result."""
        w = self._stats(winner)
        l = self._stats(loser)
        w.wins += 1
        l.losses += 1
        w.alignment += alignment
        l.alignment -= alignment
        if contradiction:
            l.reputation *= self.decay
        if not behavioral_response:
            w.reputation *= self.decay
        if cited_by:
            for src in cited_by:
                self.citation_edges.setdefault(src, {})[winner] = (
                    self.citation_edges.get(src, {}).get(winner, 0.0) + 1.0
                )
                self._stats(src).citations += 1

    # ------------------------------------------------------------------
    def _base_scores(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for name, stats in self.archetypes.items():
            total = stats.wins + stats.losses or 1
            win_rate = stats.wins / total
            rep = stats.reputation + stats.alignment + win_rate
            scores[name] = max(rep, 0.0)
        return scores

    def rankings(self) -> Dict[str, float]:
        """Return combined PageRank and reputation scores."""
        base = self._base_scores()
        ranks = pagerank(self.citation_edges)
        final: Dict[str, float] = {}
        for name in set(base) | set(ranks):
            final[name] = base.get(name, 0.0) * 0.7 + ranks.get(name, 0.0) * 0.3
        return final

    # ------------------------------------------------------------------
    def assign_user_faction(self, user_id: str, scores: Dict[str, float]) -> str:
        """Map a user to the faction with the highest alignment score."""
        if not scores:
            self.user_factions.pop(user_id, None)
            return ""
        faction = max(scores.items(), key=lambda x: x[1])[0]
        self.user_factions[user_id] = faction
        return faction


__all__ = ["SocialPowerRanker", "ArchetypeStats"]
