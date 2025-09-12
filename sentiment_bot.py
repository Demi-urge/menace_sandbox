"""Sentiment Bot for monitoring user sentiment on social media."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
from .bot_registry import BotRegistry
from .data_bot import DataBot
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

registry = BotRegistry()
data_bot = DataBot(start_server=False)

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore

from .prediction_manager_bot import PredictionManager
from .strategy_prediction_bot import StrategyPredictionBot

_POSITIVE = {"love", "great", "excellent", "good", "amazing", "awesome"}
_NEGATIVE = {"hate", "bad", "terrible", "poor", "awful", "bug"}


@dataclass
class FeedbackItem:
    """User feedback mentioning Menace or competitors."""

    text: str
    product: str
    source: str
    ts: str = datetime.utcnow().isoformat()
    sentiment: float = 0.0
    predicted: float = 0.0
    impact: float = 0.0
    profitability: float = 0.0
    label: str = "neutral"
    features: List[str] = field(default_factory=list)


class SentimentDB:
    """SQLite-backed storage for feedback items."""

    def __init__(
        self,
        path: Path | str = Path("sentiment.db"),
        *,
        router: DBRouter | None = None,
    ) -> None:
        self.path = Path(path)
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "sentiment", str(self.path), str(self.path)
        )
        self._init()

    def _init(self) -> None:
        conn = self.router.get_connection("feedback")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product TEXT,
                text TEXT,
                source TEXT,
                sentiment REAL,
                predicted REAL,
                impact REAL,
                profitability REAL,
                label TEXT,
                features TEXT,
                ts TEXT
            )
            """
        )
        conn.commit()

    def add(self, item: FeedbackItem) -> int:
        conn = self.router.get_connection("feedback")
        cur = conn.execute(
            """
            INSERT INTO feedback(product, text, source, sentiment, predicted, impact, profitability, label, features, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.product,
                item.text,
                item.source,
                item.sentiment,
                item.predicted,
                item.impact,
                item.profitability,
                item.label,
                ",".join(item.features),
                item.ts,
            ),
        )
        conn.commit()
        return cur.lastrowid

    def fetch(self, product: str | None = None, limit: int = 50) -> List[FeedbackItem]:
        conn = self.router.get_connection("feedback")
        if product:
            rows = conn.execute(
                "SELECT product, text, source, sentiment, predicted, impact, profitability, label, features, ts"
                " FROM feedback WHERE product=? ORDER BY id DESC LIMIT ?",
                (product, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT product, text, source, sentiment, predicted, impact, profitability, label, features, ts"
                " FROM feedback ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        items: List[FeedbackItem] = []
        for r in rows:
            items.append(
                FeedbackItem(
                    product=r[0],
                    text=r[1],
                    source=r[2],
                    sentiment=r[3],
                    predicted=r[4],
                    impact=r[5],
                    profitability=r[6],
                    label=r[7],
                    features=r[8].split(",") if r[8] else [],
                    ts=r[9],
                )
            )
        return items

    def avg_sentiment(self, product: str, window: int = 5) -> float:
        conn = self.router.get_connection("feedback")
        rows = conn.execute(
            "SELECT sentiment FROM feedback WHERE product=? ORDER BY id DESC LIMIT ?",
            (product, window),
        ).fetchall()
        if not rows:
            return 0.0
        return float(sum(r[0] for r in rows) / len(rows))


def classify_sentiment(text: str) -> float:
    """Return sentiment score for ``text``.

    When spaCy is available we attempt to use any built-in sentiment
    component.  Otherwise a slightly more sophisticated word based
    approach is used which takes word frequency into account.
    """

    text_lower = text.lower()
    if spacy is not None:
        try:
            nlp = spacy.load("en_core_web_sm")  # type: ignore
            doc = nlp(text_lower)
            if hasattr(doc, "_.polarity"):
                # custom sentiment extension
                return float(getattr(doc._, "polarity"))
            if getattr(doc, "sentiment", None) not in (None, 0.0):
                return float(doc.sentiment)
        except Exception:  # pragma: no cover - optional
            logger.exception("spacy sentiment failed")

    words = [w.strip(".,!?") for w in text_lower.split()]
    positives = sum(w in _POSITIVE for w in words)
    negatives = sum(w in _NEGATIVE for w in words)
    score = (positives - negatives) / max(len(words), 1)
    return float(score)


def label_from_score(score: float) -> str:
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def extract_features(text: str) -> List[str]:
    """Extract basic product features from text."""
    if spacy is not None:
        try:
            nlp = spacy.load("en_core_web_sm")  # type: ignore
        except Exception:
            nlp = spacy.blank("en")  # type: ignore
        doc = nlp(text)
        ents = [ent.text for ent in getattr(doc, "ents", [])]
        if ents:
            return ents
    words = [w.strip(".,").lower() for w in text.split() if w.isalpha()]
    return words[:3]


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class SentimentBot:
    """Collect and analyse user sentiment from social media."""

    prediction_profile = {"scope": ["sentiment"], "risk": ["low"]}

    def __init__(
        self,
        db: SentimentDB | None = None,
        *,
        prediction_manager: "PredictionManager" | None = None,
        strategy_bot: "StrategyPredictionBot" | None = None,
        ) -> None:
        self.db = db or SentimentDB()
        self.prediction_manager = prediction_manager
        self.strategy_bot = strategy_bot
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)

    def _apply_prediction_bots(self, base: float, item: FeedbackItem) -> float:
        """Combine predictions from assigned bots for future sentiment."""
        if not self.prediction_manager:
            return base
        score = base
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                other = pred([item.sentiment, float(len(item.features))])
                if isinstance(other, (list, tuple)):
                    other = other[0]
                score = (score + float(other)) / 2.0
            except Exception:
                continue
        return float(score)

    def predict_sentiment(self, item: FeedbackItem) -> float:
        """Predict future sentiment for a feedback item."""
        base = item.sentiment
        if self.prediction_manager:
            base = self._apply_prediction_bots(base, item)
        return float(base)

    def predict_profitability(self, item: FeedbackItem) -> float:
        """Predict niche profitability using assigned prediction bots."""
        if not self.prediction_manager:
            return 0.0
        score = 0.0
        count = 0
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                other = pred(item.features)
                if isinstance(other, (list, tuple)):
                    other = other[0]
                score += float(other)
                count += 1
            except Exception:
                continue
        return float(score / count) if count else 0.0

    def fetch_posts(self, urls: Iterable[str]) -> List[FeedbackItem]:
        posts: List[FeedbackItem] = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
            except Exception:
                continue
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except Exception:
                continue
            for item in data.get("items", []):
                posts.append(
                    FeedbackItem(
                        text=str(item.get("text", "")),
                        product=str(item.get("product", "")),
                        source=url,
                    )
                )
        return posts

    def analyse(self, posts: Iterable[FeedbackItem]) -> List[FeedbackItem]:
        analysed: List[FeedbackItem] = []
        for p in posts:
            p.sentiment = classify_sentiment(p.text)
            p.label = label_from_score(p.sentiment)
            p.features = extract_features(p.text)
            p.predicted = self.predict_sentiment(p)
            p.impact = p.predicted - p.sentiment
            p.profitability = self.predict_profitability(p)
            analysed.append(p)
        return analysed

    def store(self, items: Iterable[FeedbackItem]) -> None:
        for it in items:
            self.db.add(it)

    def process(self, urls: Iterable[str]) -> List[FeedbackItem]:
        posts = self.fetch_posts(urls)
        analysed = self.analyse(posts)
        self.store(analysed)
        if self.strategy_bot:
            for item in analysed:
                try:
                    self.strategy_bot.receive_sentiment(item)
                except Exception:
                    logger.exception("strategy bot receive failed")
        return analysed

    def detect_shift(self, product: str, threshold: float = 0.5, window: int = 3) -> bool:
        """Detect significant sentiment change for a product."""
        recent = self.db.fetch(product, limit=window * 2)
        if len(recent) < window * 2:
            return False
        last = recent[:window]
        prev = recent[window: window * 2]
        last_avg = sum(r.sentiment for r in last) / window
        prev_avg = sum(r.sentiment for r in prev) / window
        return abs(last_avg - prev_avg) > threshold


__all__ = [
    "FeedbackItem",
    "SentimentDB",
    "classify_sentiment",
    "label_from_score",
    "extract_features",
    "predict_sentiment",
    "predict_profitability",
    "SentimentBot",
]
