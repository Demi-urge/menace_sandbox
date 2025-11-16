from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MemoryRecord:
    """Single conversation message."""

    timestamp: float
    role: str
    content: str


@dataclass
class Preference:
    """Preference weight with confidence."""

    score: float = 0.0
    confidence: float = 0.0


class MongoMemorySystem:
    """Simple Mongo-like memory with short and long term storage."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.short_term: Dict[str, List[MemoryRecord]] = {}
        self.long_term: Dict[str, List[MemoryRecord]] = {}
        self.freq: Dict[str, Dict[str, int]] = {}
        self.preferences: Dict[str, Dict[str, Preference]] = {}
        self.archetype_graph: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    def add_message(self, user_id: str, role: str, content: str) -> None:
        now = time.time()
        rec = MemoryRecord(now, role, content)
        self.short_term.setdefault(user_id, []).append(rec)
        self.long_term.setdefault(user_id, []).append(rec)
        for tok in content.lower().split():
            self.freq.setdefault(user_id, {}).setdefault(tok, 0)
            self.freq[user_id][tok] += 1
        self._update_preferences(user_id)
        self._prune(user_id)

    # ------------------------------------------------------------------
    def _update_preferences(self, user_id: str) -> None:
        counts = self.freq.get(user_id, {})
        total = sum(counts.values()) or 1
        prefs = self.preferences.setdefault(user_id, {})
        for tok, c in counts.items():
            pref = prefs.setdefault(tok, Preference())
            weight = c / total
            pref.score = weight
            pref.confidence = min(1.0, pref.confidence + 0.1)

    # ------------------------------------------------------------------
    def inferred_preferences(self, user_id: str, defaults: List[str]) -> Dict[str, Preference]:
        prefs = self.preferences.setdefault(user_id, {})
        for key in defaults:
            prefs.setdefault(key, Preference(confidence=0.1))
        return prefs

    # ------------------------------------------------------------------
    def _prune(self, user_id: str) -> None:
        cutoff = time.time() - self.ttl_seconds
        msgs = self.short_term.get(user_id, [])
        self.short_term[user_id] = [m for m in msgs if m.timestamp >= cutoff]

    # ------------------------------------------------------------------
    def recent_messages(self, user_id: str, limit: int = 5) -> List[MemoryRecord]:
        self._prune(user_id)
        return self.short_term.get(user_id, [])[-limit:]

    # ------------------------------------------------------------------
    def update_archetype_relation(self, a1: str, a2: str, weight: float) -> None:
        g = self.archetype_graph.setdefault(a1, {})
        g[a2] = g.get(a2, 0.0) + weight
        g2 = self.archetype_graph.setdefault(a2, {})
        g2[a1] = g2.get(a1, 0.0) + weight

    def get_relation(self, a1: str, a2: str) -> float:
        return self.archetype_graph.get(a1, {}).get(a2, 0.0)


__all__ = ["MongoMemorySystem", "MemoryRecord", "Preference"]
