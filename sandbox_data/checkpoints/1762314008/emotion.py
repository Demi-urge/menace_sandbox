from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple, Optional
import os

from .sentiment import SentimentAnalyzer


@dataclass
class EmotionResult:
    """Labelled emotional state for one text."""

    timestamp: float
    primary: str
    secondary: str
    intensity: float
    confidence: float


class EmotionLabeler:
    """Simple affect recogniser using heuristics."""

    def __init__(self) -> None:
        self.analyser = SentimentAnalyzer()
        self.emotion_map = {
            "joy": "positive",
            "anger": "negative",
            "sadness": "negative",
        }

    def label(self, text: str) -> EmotionResult:
        score, emotions = self.analyser.analyse(text)
        primary = emotions[0] if emotions else "neutral"
        secondary = emotions[1] if len(emotions) > 1 else primary
        confidence = min(1.0, 0.5 + abs(score) / 2)
        return EmotionResult(
            timestamp=time.time(),
            primary=primary,
            secondary=secondary,
            intensity=score,
            confidence=confidence,
        )


class RollingEmotionTensor:
    """Maintain last five emotion readings as a tensor."""

    def __init__(self, window: int = 5) -> None:
        from collections import deque

        self.window = window
        self.readings: Deque[EmotionResult] = deque(maxlen=window)

    def update(self, result: EmotionResult) -> None:
        self.readings.append(result)

    def micro_swing(self) -> float:
        """Return the change in intensity between oldest and newest."""
        if len(self.readings) < 2:
            return 0.0
        return self.readings[-1].intensity - self.readings[0].intensity


class EmotionMemory:
    """Long-horizon emotion memory per persona."""

    def __init__(self) -> None:
        self.memory: Dict[str, List[EmotionResult]] = {}

    def log(self, persona: str, result: EmotionResult) -> None:
        self.memory.setdefault(persona, []).append(result)

    def average_intensity(self, persona: str) -> float:
        recs = self.memory.get(persona, [])
        if not recs:
            return 0.0
        return sum(r.intensity for r in recs) / len(recs)


class DatabaseEmotionMemory(EmotionMemory):
    """Persistent emotion memory backed by a SQL database."""

    def __init__(
        self,
        *,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory

    def log(self, persona: str, result: EmotionResult) -> None:  # type: ignore[override]
        from .sql_db import EmotionEntry

        Session = self.session_factory
        with Session() as s:
            s.add(
                EmotionEntry(
                    persona=persona,
                    label=result.primary,
                    intensity=result.intensity,
                    timestamp=result.timestamp,
                )
            )
            s.commit()

    def average_intensity(self, persona: str) -> float:  # type: ignore[override]
        from .sql_db import EmotionEntry

        Session = self.session_factory
        with Session() as s:
            recs = s.query(EmotionEntry).filter_by(persona=persona).all()
        if not recs:
            return 0.0
        return sum(r.intensity for r in recs) / len(recs)


class GenderStyleAdapter:
    """Infer gender presentation and apply style filters."""

    def __init__(self) -> None:
        self.opt_out: Dict[str, bool] = {}

    def infer_gender(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in {"she", "her", "woman"}):
            return "female"
        if any(w in lower for w in {"he", "him", "man"}):
            return "male"
        return "unknown"

    def adapt(self, user_id: str, text: str) -> str:
        if self.opt_out.get(user_id):
            return text
        gender = self.infer_gender(text)
        if gender == "female":
            return text.replace("you", "ma'am")
        if gender == "male":
            return text.replace("you", "sir")
        return text

    def set_opt_out(self, user_id: str, value: bool) -> None:
        self.opt_out[user_id] = value


class ReinforcementABTest:
    """Track dwell time, sentiment delta and participation."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[Tuple[float, float, int]]] = {"A": [], "B": []}

    def log(self, variant: str, dwell: float, delta: float, participation: int) -> None:
        self.metrics.setdefault(variant, []).append((dwell, delta, participation))

    def results(self, variant: str) -> Tuple[float, float, float]:
        data = self.metrics.get(variant, [])
        if not data:
            return (0.0, 0.0, 0.0)
        avg_dwell = sum(d[0] for d in data) / len(data)
        avg_delta = sum(d[1] for d in data) / len(data)
        avg_part = sum(d[2] for d in data) / len(data)
        return avg_dwell, avg_delta, avg_part
