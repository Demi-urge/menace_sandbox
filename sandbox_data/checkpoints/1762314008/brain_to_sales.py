from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .neuro_etl import NeuroToken
from .trigger_phrase_db import TriggerPhraseDB

PERSUASION_SCRIPTS = {
    "scarcity": "Limited-Time Offer",
    "social_proof": "Join the Crowd",
    "loss_aversion": "Risk-Free Trial",
}

@dataclass
class BrainToSalesMapper:
    """Map neural activation tokens to persuasion scripts."""

    phrase_db: TriggerPhraseDB
    region_index: Dict[str, Set[str]] = field(default_factory=dict)
    expected_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    script_map: Dict[str, str] = field(default_factory=lambda: PERSUASION_SCRIPTS)

    # --------------------------------------------------------------
    def process_token(self, token: NeuroToken, archetype: str) -> None:
        script = self.script_map.get(token.primitive)
        if not script or not token.region:
            return
        # link region to script
        self.region_index.setdefault(token.region, set()).add(script)
        self.phrase_db.add_phrase(script, [token.region])
        ev = self.expected_values.setdefault(script, {})
        ev[archetype] = ev.get(archetype, 0.0) + token.effect_size

    # --------------------------------------------------------------
    def process_tokens(self, tokens: List[NeuroToken], archetype: str) -> None:
        for t in tokens:
            self.process_token(t, archetype)

    # --------------------------------------------------------------
    def scripts_for_region(self, region: str) -> List[str]:
        return list(self.region_index.get(region, []))

    def expected_value(self, script: str, archetype: str) -> float:
        return self.expected_values.get(script, {}).get(archetype, 0.0)
