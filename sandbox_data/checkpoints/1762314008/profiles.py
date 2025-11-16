from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Profile:
    """User profile storing interaction features and preference scores."""

    emotional_tone: float = 0.0
    attention_retention_time: float = 0.0
    response_type: str = ""
    dopamine_indicator: float = 0.0
    preference_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class ProfileManager:
    """Manage user profiles with dynamic scoring and decay."""

    def __init__(self, decay_factor: float = 0.9) -> None:
        self.decay_factor = decay_factor
        self.profiles: Dict[str, Profile] = {}

    def _update_score(self, profile: Profile, key: str, value: float) -> None:
        current = profile.preference_scores.get(key, 0.0)
        profile.preference_scores[key] = current * self.decay_factor + value

    def update_profile(
        self,
        user_id: str,
        emotional_tone: float,
        attention_time: float,
        response_type: str,
        dopamine_indicator: float,
    ) -> None:
        profile = self.profiles.get(user_id, Profile())
        profile.emotional_tone = emotional_tone
        profile.attention_retention_time = attention_time
        profile.response_type = response_type
        profile.dopamine_indicator = dopamine_indicator

        self._update_score(profile, "emotional_tone", emotional_tone)
        self._update_score(profile, "attention_retention_time", attention_time)
        self._update_score(profile, "dopamine_indicator", dopamine_indicator)

        profile.last_updated = time.time()
        self.profiles[user_id] = profile

    def decay_profiles(self) -> None:
        for profile in self.profiles.values():
            for key, value in profile.preference_scores.items():
                profile.preference_scores[key] = value * self.decay_factor
