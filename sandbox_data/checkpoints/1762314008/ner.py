from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Iterable

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except Exception:  # pragma: no cover - optional heavy dep
    spacy = None
    PhraseMatcher = None  # type: ignore

# Default phrase banks for intent, emotion, and context cues
DEFAULT_PHRASE_BANK: Dict[str, List[str]] = {
    "intent": ["buy", "want", "need"],
    "emotion": ["nervous", "curious", "skeptical"],
    "context": ["reddit", "student", "crypto", "instagram"],
}

# Mapping of raw phrases to brain region style trigger labels
TRIGGER_LABEL_MAP: Dict[str, str] = {
    "group chat": "tpj_tribe_appeal",
    "limited offer": "amygdala_urgency",
}

# Aliases for standardisation
ALIAS_MAP: Dict[str, str] = {
    "ig": "instagram",
}


@dataclass
class IntentProfile:
    """Temporary profile storing trigger counts within an interaction window."""

    counts: Dict[str, int] = field(default_factory=dict)
    last_seen: Dict[str, float] = field(default_factory=dict)


class IntentEntityExtractor:
    """Extract intent-related entities using spaCy phrase matching."""

    def __init__(
        self,
        phrase_bank: Dict[str, List[str]] | None = None,
        alias_map: Dict[str, str] | None = None,
        trigger_map: Dict[str, str] | None = None,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.phrase_bank = phrase_bank or DEFAULT_PHRASE_BANK
        self.alias_map = alias_map or ALIAS_MAP
        self.trigger_map = trigger_map or TRIGGER_LABEL_MAP
        self.ttl_seconds = ttl_seconds
        self.profile = IntentProfile()

        if spacy is not None and PhraseMatcher is not None:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        else:  # fallback simple matching
            self.nlp = None
            self.matcher = None
        patterns = set()
        for phrases in self.phrase_bank.values():
            for p in phrases:
                patterns.add(p)
                for alias, canonical in self.alias_map.items():
                    if canonical == p:
                        patterns.add(alias)
        if self.matcher is not None:
            self.matcher.add("TRIGGER", [self.nlp.make_doc(p) for p in patterns])
        
        if self.trigger_map:
            label_patterns = set()
            for p in self.trigger_map.keys():
                label_patterns.add(p)
                for alias, canonical in self.alias_map.items():
                    if canonical == p:
                        label_patterns.add(alias)
            if self.matcher is not None:
                self.matcher.add(
                    "TRIGGER_LABELS", [self.nlp.make_doc(p) for p in label_patterns]
                )

    def _standardize(self, text: str) -> str:
        return self.alias_map.get(text.lower(), text.lower())

    def _map_label(self, phrase: str) -> str:
        phrase = phrase.lower()
        phrase = self.alias_map.get(phrase, phrase)
        return self.trigger_map.get(phrase, phrase)

    def _prune(self) -> None:
        now = time.time()
        to_remove = [
            k
            for k, t in self.profile.last_seen.items()
            if now - t > self.ttl_seconds and self.profile.counts.get(k, 0) < 2
        ]
        for k in to_remove:
            self.profile.counts.pop(k, None)
            self.profile.last_seen.pop(k, None)

    def extract(self, text: str) -> List[str]:
        triggers: List[str] = []
        now = time.time()
        if self.matcher is not None:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            for _, start, end in matches:
                phrase = doc[start:end].text
                label = self._map_label(phrase)
                triggers.append(label)
                self.profile.counts[label] = self.profile.counts.get(label, 0) + 1
                self.profile.last_seen[label] = now
        else:
            text_low = text.lower()
            for phrase in set(sum(self.phrase_bank.values(), [])):
                variants = {phrase}
                for alias, canonical in self.alias_map.items():
                    if canonical == phrase:
                        variants.add(alias)
                for var in variants:
                    if var.lower() in text_low:
                        label = self._map_label(var)
                        triggers.append(label)
                        self.profile.counts[label] = self.profile.counts.get(label, 0) + 1
                        self.profile.last_seen[label] = now
        self._prune()
        return triggers

    def get_intent_profile(self) -> Dict[str, int]:
        """Return current trigger counts."""
        self._prune()
        return dict(self.profile.counts)
