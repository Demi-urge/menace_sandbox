from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
from .metrics import metrics


@dataclass
class PhraseStats:
    """Store performance metrics and context counts for a trigger phrase."""

    phrase: str
    brain_regions: List[str]
    clicks: int = 0
    replies: int = 0
    conversions: int = 0
    context_counts: Dict[str, int] = field(default_factory=dict)


class TriggerPhraseDB:
    """Dynamic database of trigger phrases with reinforcement-style scoring."""

    def __init__(self) -> None:
        self.phrases: Dict[str, PhraseStats] = {}

    def add_phrase(self, phrase: str, brain_regions: List[str]) -> None:
        """Add a phrase with brain-region tags if not already present."""
        if phrase not in self.phrases:
            self.phrases[phrase] = PhraseStats(phrase, list(brain_regions))

    def record_feedback(
        self,
        phrase: str,
        *,
        context: str,
        clicks: int = 0,
        replies: int = 0,
        conversions: int = 0,
    ) -> None:
        """Update metrics for a phrase within a specific context."""
        stats = self.phrases.setdefault(phrase, PhraseStats(phrase, []))
        stats.clicks += clicks
        stats.replies += replies
        stats.conversions += conversions
        if conversions:
            metrics.record_conversion(conversions)
        stats.context_counts[context] = stats.context_counts.get(context, 0) + 1

    def score(self, phrase: str, *, context: str | None = None) -> float:
        """Return a weighted score based on metrics and optional context."""
        stats = self.phrases.get(phrase)
        if not stats:
            return 0.0
        base = stats.conversions * 3 + stats.replies * 2 + stats.clicks
        if context:
            base += stats.context_counts.get(context, 0)
        return base

    def best_phrase(self, phrases: List[str], *, context: str) -> str:
        """Return the phrase with the highest score for a context."""
        if not phrases:
            return ""
        return max(phrases, key=lambda p: self.score(p, context=context))

    def prune(self, threshold: float) -> None:
        """Remove phrases scoring below the threshold."""
        to_remove = [p for p in self.phrases if self.score(p) < threshold]
        for p in to_remove:
            self.phrases.pop(p, None)

    def mutate(self, top_n: int = 3) -> List[str]:
        """Return simple mutated variations of top phrases."""
        ranked = sorted(self.phrases.values(), key=lambda s: self.score(s.phrase), reverse=True)
        top = ranked[:top_n]
        variations = [f"{s.phrase}!" for s in top]
        return variations
