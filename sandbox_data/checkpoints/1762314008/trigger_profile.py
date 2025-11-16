from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TriggerProfile:
    """Store cumulative trigger scores for a user."""

    scores: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class TriggerProfileScorer:
    """Evaluate interactions and update trigger profiles."""

    def __init__(self, categories: List[str]) -> None:
        self.categories = categories
        self.profiles: Dict[str, TriggerProfile] = {}

    def score_interaction(self, user_id: str, triggers: Dict[str, float]) -> str:
        """Update profile with trigger values and return dominant category."""
        profile = self.profiles.get(user_id, TriggerProfile())
        for cat in self.categories:
            value = triggers.get(cat, 0.0)
            profile.scores[cat] = profile.scores.get(cat, 0.0) + value
        profile.last_updated = time.time()
        self.profiles[user_id] = profile

        if not triggers:
            return ""
        best_cat = max(triggers.items(), key=lambda x: x[1])[0]
        return best_cat

    def get_profile(self, user_id: str) -> TriggerProfile:
        return self.profiles.get(user_id, TriggerProfile())
