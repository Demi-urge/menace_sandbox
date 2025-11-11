from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - optional heavy dep
    SentimentIntensityAnalyzer = None

try:
    import spacy
except Exception:  # pragma: no cover - optional heavy dep
    spacy = None


@dataclass
class SentimentRecord:
    timestamp: float
    score: float
    emotions: List[str]


class SentimentAnalyzer:
    """Basic sentiment and emotion analyzer with intensity adjustment."""

    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        if spacy is not None:
            print("[DEBUG] Current PATH during spacy load:", os.environ["PATH"])
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None
        self.intensifiers = {"very", "extremely", "really", "so", "super"}
        self.positive_words = {"happy", "joy", "excited"}
        self.negative_words = {"sad", "angry", "mad", "furious"}

    def analyse(self, text: str) -> Tuple[float, List[str]]:
        if self.nlp is not None:
            doc = self.nlp(text)
        else:  # simple token objects with lemma attr
            class Tok:
                def __init__(self, t: str) -> None:
                    self.text = t
                    self.lemma_ = t.strip(".,!?\"'").lower()

            doc = [Tok(t) for t in text.split()]
        if self.vader:
            score = self.vader.polarity_scores(text)["compound"]
        else:  # simple fallback
            score = 0.0
            for token in doc:
                lower = token.lemma_.lower()
                if lower in self.positive_words:
                    score += 0.5
                if lower in self.negative_words:
                    score -= 0.5
            score = max(-1.0, min(1.0, score))

        # adjust for intensifiers
        modifier = 0
        for token in doc:
            if token.lemma_.lower() in self.intensifiers:
                modifier += 0.2
        if "!" in text:
            modifier += 0.1
        score = max(-1.0, min(1.0, score * (1 + modifier)))

        emotions: List[str] = []
        for token in doc:
            lower = token.lemma_.lower()
            if lower in self.positive_words:
                emotions.append("joy")
            if lower in self.negative_words:
                emotions.append("anger" if lower in {"angry", "mad", "furious"} else "sadness")
        return score, sorted(set(emotions))


class SentimentMemory:
    """Store sentiment scores per user and track global trends."""

    def __init__(self) -> None:
        self.user_history: Dict[str, List[SentimentRecord]] = {}
        self.global_monthly: Dict[str, List[float]] = {}

    def log(self, user_id: str, score: float, emotions: List[str], *, timestamp: Optional[float] = None) -> None:
        ts = time.time() if timestamp is None else timestamp
        record = SentimentRecord(timestamp=ts, score=score, emotions=emotions)
        self.user_history.setdefault(user_id, []).append(record)
        month = time.strftime("%Y-%m", time.localtime(ts))
        self.global_monthly.setdefault(month, []).append(score)

    def average_user_sentiment(self, user_id: str) -> float:
        records = self.user_history.get(user_id, [])
        if not records:
            return 0.0
        return sum(r.score for r in records) / len(records)

    def global_trend(self, month: str) -> float:
        scores = self.global_monthly.get(month, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
