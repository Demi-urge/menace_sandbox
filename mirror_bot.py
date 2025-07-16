"""Mirror Bot for logging conversations and mimicking user style."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

_POSITIVE = {"great", "good", "love", "excellent", "nice", "awesome"}
_NEGATIVE = {"bad", "terrible", "hate", "awful", "poor"}


def sentiment_score(text: str) -> float:
    text_lower = text.lower()
    score = sum(w in text_lower for w in _POSITIVE) - sum(w in text_lower for w in _NEGATIVE)
    return float(score) / max(len(_POSITIVE), 1)


@dataclass
class InteractionRecord:
    user: str
    response: str
    feedback: str = ""
    sentiment: float = 0.0
    ts: str = datetime.utcnow().isoformat()


class MirrorDB:
    def __init__(self, path: Path | str = Path("mirror.db")) -> None:
        self.path = Path(path)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT,
                    response TEXT,
                    feedback TEXT,
                    sentiment REAL,
                    ts TEXT
                )
                """
            )
            conn.commit()

    def add(self, rec: InteractionRecord) -> int:
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                "INSERT INTO logs(user, response, feedback, sentiment, ts) VALUES (?, ?, ?, ?, ?)",
                (rec.user, rec.response, rec.feedback, rec.sentiment, rec.ts),
            )
            conn.commit()
            return cur.lastrowid

    def fetch(self, limit: int = 100) -> List[InteractionRecord]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT user, response, feedback, sentiment, ts FROM logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [InteractionRecord(*r) for r in rows]

    def avg_sentiment(self, last: int = 5) -> float:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT sentiment FROM logs ORDER BY id DESC LIMIT ?",
                (last,),
            ).fetchall()
        if not rows:
            return 0.0
        return float(sum(r[0] for r in rows) / len(rows))


class MirrorBot:
    """Collect user interactions and mirror communication style."""

    def __init__(self, db: MirrorDB | None = None) -> None:
        self.db = db or MirrorDB()
        self.styles: List[str] = []

    def log_interaction(self, user: str, response: str, feedback: str | None = None) -> int:
        score = sentiment_score(feedback or response)
        rec = InteractionRecord(user=user, response=response, feedback=feedback or "", sentiment=score)
        return self.db.add(rec)

    def update_style(self, suggestion: str) -> None:
        self.styles.append(suggestion)

    def generate_response(self, user_text: str) -> str:
        base = user_text
        if self.styles:
            base = f"{base} {self.styles[-1]}"
        if self.db.avg_sentiment() > 0:
            base = f"{base} :)"
        return base

    def history(self, limit: int = 20) -> List[InteractionRecord]:
        return self.db.fetch(limit)

__all__ = [
    "InteractionRecord",
    "MirrorDB",
    "MirrorBot",
    "sentiment_score",
]
