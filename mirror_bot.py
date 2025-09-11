"""Mirror Bot for logging conversations and mimicking user style."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass
from datetime import datetime
from typing import List

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

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
    def __init__(self, *, router: DBRouter | None = None) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router("mirror")
        self.conn = self.router.get_connection("mirror_logs")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                response TEXT,
                feedback TEXT,
                sentiment REAL,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: InteractionRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO mirror_logs(user, response, feedback, sentiment, ts) VALUES (?, ?, ?, ?, ?)",
            (rec.user, rec.response, rec.feedback, rec.sentiment, rec.ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def fetch(self, limit: int = 100) -> List[InteractionRecord]:
        rows = self.conn.execute(
            "SELECT user, response, feedback, sentiment, ts FROM mirror_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [InteractionRecord(*r) for r in rows]

    def avg_sentiment(self, last: int = 5) -> float:
        rows = self.conn.execute(
            "SELECT sentiment FROM mirror_logs ORDER BY id DESC LIMIT ?",
            (last,),
        ).fetchall()
        if not rows:
            return 0.0
        return float(sum(r[0] for r in rows) / len(rows))


@self_coding_managed
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
