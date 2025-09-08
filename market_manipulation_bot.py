from __future__ import annotations

"""Bot for aggressive niche saturation to outcompete rivals (legal only)."""

from typing import Iterable, Tuple
import time

from .competitive_intelligence_bot import CompetitiveIntelligenceBot
from .niche_saturation_bot import NicheCandidate, NicheSaturationBot
from vector_service.context_builder import ContextBuilder
from .compliance_checker import ComplianceChecker


class MarketManipulationBot:
    """
    Saturate market niches using high-effort content and competitive targeting.
    Designed to increase visibility and engagement within underserved spaces.
    """

    def __init__(
        self,
        intel_bot: CompetitiveIntelligenceBot | None = None,
        saturation_bot: NicheSaturationBot | None = None,
        *,
        checker: ComplianceChecker | None = None,
        role: str = "trader",
        context_builder: ContextBuilder,
    ) -> None:
        self.intel_bot = intel_bot or CompetitiveIntelligenceBot()
        context_builder.refresh_db_weights()
        self.context_builder = context_builder
        self.saturation_bot = (
            saturation_bot
            if saturation_bot is not None
            else NicheSaturationBot(context_builder=self.context_builder)
        )
        self.checker = checker or ComplianceChecker()
        self.role = role

    def saturate(self, niches: Iterable[str]) -> list[Tuple[str, bool]]:
        """Analyse competitive landscape and saturate the given niches."""
        niches_l = list(niches)
        try:
            self.context_builder.refresh_db_weights()
            sources = ("bots", "code", "errors", "workflows")
            self._pre_saturation_ctx = {
                src: self.context_builder.build_context(src, top_k=1)
                for src in sources
            }
        except Exception:
            self._pre_saturation_ctx = {}
        if self.checker:
            if not self.checker.verify_permission(self.role, "trade"):
                return []
            if not self.checker.check_trade({"volume": len(niches_l)}):
                return []
        updates = self.intel_bot.db.fetch()
        candidates: list[NicheCandidate] = []
        for n in niches_l:
            text_matches = [
                u for u in updates if n.lower() in (u.title + " " + u.content).lower()
            ]
            if text_matches:
                pos = sum(u.sentiment for u in text_matches if u.sentiment > 0)
                neg = sum(-u.sentiment for u in text_matches if u.sentiment < 0)
                recency = [1 / (1 + (time.time() - u.timestamp) / 86400) for u in text_matches]
                demand = max(sum(recency) + pos, 1.0)
                competition = max((neg + sum(1 - r for r in recency)) / len(text_matches), 0.1)
                trend = sum(recency) / len(text_matches)
            else:
                demand = 1.0
                competition = 1.0
                trend = 0.0
            candidates.append(
                NicheCandidate(name=n, demand=demand, competition=competition, trend=trend)
            )
        return self.saturation_bot.saturate(candidates)


__all__ = ["MarketManipulationBot"]
