from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

from .sentiment import SentimentAnalyzer
from .user_preferences import PreferenceProfile


@dataclass
class CandidateStats:
    success: int = 1
    fail: int = 1


class ConfidenceBasedActionSelector:
    """Assign styles based on certainty derived from multiple signals."""

    def __init__(self, base_fire: float = 0.8, base_soften: float = 0.5) -> None:
        self.base_fire = base_fire
        self.base_soften = base_soften
        self.sentiment = SentimentAnalyzer()
        self.history: Dict[str, CandidateStats] = {}
        self.user_mod: Dict[str, float] = {}

    # ---------------------- util ----------------------
    def _embed(self, text: str) -> Tuple[float, ...]:
        from .embedding import embed_text

        return tuple(embed_text(text))

    def _cosine(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        denom = norm_a * norm_b
        return dot / denom if denom else 0.0

    def _win_rate(self, cand: str) -> float:
        stats = self.history.get(cand)
        if not stats:
            return 0.5
        total = stats.success + stats.fail
        return stats.success / total if total else 0.5

    # ---------------------- scoring ----------------------
    def certainty(self, message: str, candidate: str, profile: PreferenceProfile) -> float:
        msg_emb = self._embed(message)
        cand_emb = self._embed(candidate)
        sem = self._cosine(msg_emb, cand_emb)
        user_sent, _ = self.sentiment.analyse(message)
        cand_sent, _ = self.sentiment.analyse(candidate)
        sent_align = 1.0 - abs(user_sent - cand_sent) / 2
        win = self._win_rate(candidate)
        if profile.embedding:
            user_emb = tuple(profile.embedding[: len(cand_emb)])
            personal = self._cosine(user_emb, cand_emb)
        else:
            personal = 0.5
        certainty = (sem + sent_align + win + personal) / 4
        return max(0.0, min(1.0, certainty))

    def record_feedback(self, candidate: str, success: bool) -> None:
        stats = self.history.setdefault(candidate, CandidateStats())
        if success:
            stats.success += 1
        else:
            stats.fail += 1

    # ---------------------- thresholds ----------------------
    def _thresholds(self, user_id: str) -> Tuple[float, float]:
        mod = self.user_mod.get(user_id, 0.0)
        fire = max(0.0, min(1.0, self.base_fire + mod))
        soft = max(0.0, min(1.0, self.base_soften + mod))
        return fire, soft

    def adjust_user(self, user_id: str, *, corrected: bool) -> None:
        mod = self.user_mod.get(user_id, 0.0)
        mod += 0.02 if corrected else -0.01
        self.user_mod[user_id] = max(-0.2, min(0.2, mod))

    # ---------------------- style decision ----------------------
    def select_style(self, user_id: str, message: str, candidate: str, profile: PreferenceProfile) -> Tuple[str, float]:
        cert = self.certainty(message, candidate, profile)
        fire, soft = self._thresholds(user_id)
        if cert >= fire:
            return "fire", cert
        if cert >= soft:
            return "velvet", cert
        return "intel", cert

    def disclaimer(self, text: str, certainty: float) -> str:
        if certainty < self.base_soften:
            return f"Could be off here—help me out? {text}"
        if certainty < self.base_fire:
            return f"{text} (I think)"
        return text

