from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import os


@dataclass
class PreferenceRecord:
    """Store a single message with analysis information."""

    timestamp: float
    tokens: List[str]
    embedding: List[float]
    intensity: float
    contradiction: bool


@dataclass
class PreferenceProfile:
    """Aggregated preference data for a user."""

    keyword_freq: Dict[str, float] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    archetype: str = ""
    last_updated: float = field(default_factory=time.time)


class PreferenceEngine:
    """Learn user preferences with decay and basic clustering."""

    def __init__(self, window_seconds: int = 3600) -> None:
        self.window_seconds = window_seconds
        self.messages: Dict[str, Deque[PreferenceRecord]] = {}
        self.profiles: Dict[str, PreferenceProfile] = {}

    # ------------------ Utility functions ------------------
    def _simple_embed(self, text: str, dims: Optional[int] = None) -> List[float]:
        from .embedding import embed_text

        emb = embed_text(text)
        if dims is not None and len(emb) != dims:
            return emb[:dims]
        return emb

    def _tokenize(self, text: str) -> List[str]:
        return [t.strip(".,!?\"'`").lower() for t in text.split() if t]

    # ------------------ Public API ------------------
    def add_message(self, user_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        intensity = sum(1 for t in tokens if t in {"very", "extremely", "really"}) / (len(tokens) or 1)
        contradiction = any(t in {"not", "n't", "no"} for t in tokens)
        emb = self._simple_embed(text)
        record = PreferenceRecord(time.time(), tokens, emb, intensity, contradiction)
        q = self.messages.setdefault(user_id, deque())
        q.append(record)
        self._prune(user_id)
        self._recompute_profile(user_id)

    def _prune(self, user_id: str) -> None:
        q = self.messages.get(user_id)
        if not q:
            return
        expiry = time.time() - self.window_seconds
        while q and q[0].timestamp < expiry:
            q.popleft()

    def _recompute_profile(self, user_id: str) -> None:
        q = self.messages.get(user_id, deque())
        freq: Dict[str, float] = {}
        embeddings: List[List[float]] = []
        for rec in q:
            for t in rec.tokens:
                if not t.isalpha():
                    continue
                weight = 1.0 + rec.intensity
                if rec.contradiction:
                    weight *= -1.0
                freq[t] = freq.get(t, 0.0) + weight
            embeddings.append(rec.embedding)
        if embeddings:
            dim = len(embeddings[0])
            avg = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
        else:
            avg = []
        profile = self.profiles.get(user_id, PreferenceProfile())
        profile.keyword_freq = freq
        profile.embedding = avg
        profile.last_updated = time.time()
        self.profiles[user_id] = profile

    def get_profile(self, user_id: str) -> PreferenceProfile:
        return self.profiles.get(user_id, PreferenceProfile())

    # ------------------ Archetype clustering ------------------
    def assign_archetypes(self, k: int = 3, iterations: int = 5) -> None:
        users = list(self.profiles.keys())
        if not users:
            return
        embeddings = [self.profiles[u].embedding or [0.0] * 5 for u in users]
        dim = len(embeddings[0])
        # initialize centroids
        centroids = embeddings[:k]
        random.seed(0)
        while len(centroids) < k:
            centroids.append([random.random() for _ in range(dim)])
        assignments = [0] * len(users)
        for _ in range(iterations):
            # assign
            for idx, emb in enumerate(embeddings):
                dists = [self._dist(emb, c) for c in centroids]
                assignments[idx] = dists.index(min(dists))
            # recompute
            for ci in range(k):
                members = [embeddings[i] for i, a in enumerate(assignments) if a == ci]
                if not members:
                    continue
                centroids[ci] = [sum(v[j] for v in members) / len(members) for j in range(dim)]
        for idx, user in enumerate(users):
            arche = f"archetype_{assignments[idx]}"
            self.profiles[user].archetype = arche

    def _dist(self, a: List[float], b: List[float]) -> float:
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


@dataclass
class InteractionRecord:
    user_id: str
    archetype: str
    score: float
    explanation: Dict[str, float]


class PerformanceTracker:
    """Track how users perform with different archetypes."""

    def __init__(self) -> None:
        self.records: List[InteractionRecord] = []

    def log_interaction(self, user_id: str, archetype: str, features: Dict[str, float]) -> float:
        score = sum(features.values()) / (len(features) or 1)
        total = sum(abs(v) for v in features.values()) or 1.0
        explanation = {k: v / total for k, v in features.items()}
        self.records.append(InteractionRecord(user_id, archetype, score, explanation))
        return score

    def history(self, user_id: str) -> List[InteractionRecord]:
        return [r for r in self.records if r.user_id == user_id]


class RoleplayCoach:
    """Simulate roleplay interactions using archetypes."""

    def __init__(self, engine: PreferenceEngine, tracker: PerformanceTracker) -> None:
        self.engine = engine
        self.tracker = tracker
        self.goals = {
            "archetype_0": "be analytical",
            "archetype_1": "be motivational",
            "archetype_2": "be friendly",
        }

    def interact(self, user_id: str, message: str) -> Tuple[str, float]:
        self.engine.add_message(user_id, message)
        profile = self.engine.get_profile(user_id)
        if not profile.archetype:
            self.engine.assign_archetypes()
            profile = self.engine.get_profile(user_id)
        goal = self.goals.get(profile.archetype, "be helpful")
        response = f"[{profile.archetype}] {goal}: {message}"
        score = self.tracker.log_interaction(user_id, profile.archetype, {"length": len(message)})
        return response, score


class DatabasePreferenceEngine(PreferenceEngine):
    """Preference engine that persists records to a SQL database."""

    def __init__(
        self,
        window_seconds: int = 3600,
        *,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        super().__init__(window_seconds=window_seconds)
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory

        # load existing messages
        from .sql_db import PreferenceMessage

        Session = self.session_factory
        self.messages = {}
        self.profiles = {}
        with Session() as s:
            rows = s.query(PreferenceMessage).order_by(PreferenceMessage.timestamp).all()
        for r in rows:
            rec = PreferenceRecord(
                timestamp=r.timestamp,
                tokens=r.tokens or [],
                embedding=r.embedding or [],
                intensity=r.intensity or 0.0,
                contradiction=bool(r.contradiction),
            )
            self.messages.setdefault(r.user_id, deque()).append(rec)
        for uid in list(self.messages.keys()):
            self._prune(uid)
            self._recompute_profile(uid)

    def add_message(self, user_id: str, text: str) -> None:  # type: ignore[override]
        super().add_message(user_id, text)
        from .sql_db import PreferenceMessage

        Session = self.session_factory
        rec = self.messages[user_id][-1]
        with Session() as s:
            s.add(
                PreferenceMessage(
                    user_id=user_id,
                    timestamp=rec.timestamp,
                    tokens=rec.tokens,
                    embedding=rec.embedding,
                    intensity=rec.intensity,
                    contradiction=rec.contradiction,
                )
            )
            s.commit()
