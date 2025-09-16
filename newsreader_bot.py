"""Newsreader Bot for analysing economic and business events."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from context_builder import handle_failure, PromptBuildError

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional heavy dependency
    from .chatgpt_idea_bot import ChatGPTClient

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore

try:
    import gensim  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gensim = None  # type: ignore

try:  # canonical tag constants for logging
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # type: ignore
try:  # helper for tagging log entries
    from .memory_logging import ensure_tags
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_logging import ensure_tags  # type: ignore

logger = logging.getLogger(__name__)

from db_router import GLOBAL_ROUTER, init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope
from dynamic_path_router import resolve_path

DB_PATH = resolve_path("news.db")


@dataclass
class Event:
    """Representation of a news event."""

    title: str
    summary: str
    source: str
    timestamp: str
    categories: List[str] = field(default_factory=list)
    sentiment: float = 0.0
    exploration_depth: int = 0
    impact: float = 0.0


class NewsDB:
    """SQLite backed storage for events."""

    def __init__(self, menace_id: str = "default", path: Path = DB_PATH) -> None:
        self.path = path
        self.router = GLOBAL_ROUTER or init_db_router(menace_id)
        self._init()

    def _init(self) -> None:
        conn = self.router.get_connection("events")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                summary TEXT,
                source TEXT,
                timestamp TEXT,
                categories TEXT,
                sentiment REAL,
                exploration_depth INTEGER,
                impact REAL
            )
            """
        )
        conn.commit()

    def add(self, event: Event) -> int:
        conn = self.router.get_connection("events")
        cur = conn.execute(
            """
            INSERT INTO events
                (title, summary, source, timestamp, categories, sentiment,
                 exploration_depth, impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.title,
                event.summary,
                event.source,
                event.timestamp,
                ",".join(event.categories),
                event.sentiment,
                event.exploration_depth,
                event.impact,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def fetch(
        self, limit: int = 100, *, scope: Scope | str = "local"
    ) -> List[Event]:
        """Return stored events filtered by menace ``scope``."""

        conn = self.router.get_connection("events")
        base = (
            "SELECT title, summary, source, timestamp, categories, sentiment,"
            " exploration_depth, impact FROM events"
        )
        clause, scope_params = build_scope_clause("events", scope, self.router.menace_id)
        base = apply_scope(base, clause)
        base += " ORDER BY id DESC LIMIT ?"
        params = [*scope_params, limit]
        rows = conn.execute(base, params).fetchall()
        events = []
        for row in rows:
            events.append(
                Event(
                    title=row[0],
                    summary=row[1],
                    source=row[2],
                    timestamp=row[3],
                    categories=row[4].split(",") if row[4] else [],
                    sentiment=row[5],
                    exploration_depth=row[6],
                    impact=row[7],
                )
            )
        return events

    def increment_depth(self, title: str) -> None:
        conn = self.router.get_connection("events")
        conn.execute(
            "UPDATE events SET exploration_depth = exploration_depth + 1 WHERE title = ?",
            (title,),
        )
        conn.commit()


quick_terms = [
    "quick profit",
    "fast roi",
    "low investment",
]
scale_terms = [
    "platform",
    "scalable",
    "infrastructure",
]


def fetch_news(
    api_url: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
    *,
    energy: float | None = None,
) -> List[Event]:
    """Fetch articles from a generic news API."""
    if params is None:
        params = {}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(api_url, headers=headers, params=params, timeout=10)
    except Exception:
        return []
    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return []
    articles = data.get("articles", [])
    events: List[Event] = []
    for art in articles:
        events.append(
            Event(
                title=str(art.get("title", "")),
                summary=str(art.get("description", "")),
                source=str(art.get("source", {}).get("name", "")),
                timestamp=str(art.get("publishedAt", datetime.utcnow().isoformat())),
            )
        )
    if energy is not None:
        terms = quick_terms if energy < 0.5 else scale_terms
        filtered = [
            e
            for e in events
            if any(t in f"{e.title} {e.summary}".lower() for t in terms)
        ]
        if filtered:
            events = filtered
    return events


def filter_events(events: Iterable[Event], keywords: Iterable[str]) -> List[Event]:
    """Return events containing any of the given keywords."""
    keys = [k.lower() for k in keywords]
    filtered: List[Event] = []
    for e in events:
        text = f"{e.title} {e.summary}".lower()
        if any(k in text for k in keys):
            filtered.append(e)
    return filtered


def cluster_events(events: List[Event], n_clusters: int = 2) -> List[List[Event]]:
    """Group similar events using TF-IDF and KMeans."""
    if not events:
        return []
    docs = [f"{e.title} {e.summary}" for e in events]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)
    n_clusters = min(n_clusters, len(events))
    if n_clusters == 0:
        return []
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(X)
    clusters: Dict[int, List[Event]] = {}
    for lbl, ev in zip(labels, events):
        clusters.setdefault(lbl, []).append(ev)
    return list(clusters.values())


class ImpactPredictor:
    """Predict economic impact using a linear model."""

    def __init__(self) -> None:
        self.model = LinearRegression()
        self.trained = False

    def train(self, events: Iterable[Event]) -> None:
        xs = [len(e.summary.split()) for e in events]
        ys = [e.sentiment for e in events]
        if len(xs) < 2:
            return
        X = np.array(xs).reshape(-1, 1)
        y = np.array(ys)
        self.model.fit(X, y)
        self.trained = True

    def predict(self, event: Event) -> float:
        if not self.trained:
            return 0.0
        X = np.array([[len(event.summary.split())]])
        return float(self.model.predict(X)[0])


def monetise_event(client: "ChatGPTClient", event: Event) -> str:
    """Request monetisation strategies from ChatGPT."""
    from .chatgpt_idea_bot import ChatGPTClient as _Client  # local import

    if not isinstance(client, _Client):  # pragma: no cover - type check
        return ""
    prompt = "Suggest monetisation strategies for this event."
    base_tags = [FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT]
    full_tags = ensure_tags("newsreader_bot.monetise_event", base_tags)
    intent = {"title": event.title, "summary": event.summary}
    try:
        prompt_obj = client.context_builder.build_prompt(
            prompt, intent_metadata=intent, tags=full_tags
        )
    except Exception as exc:
        if isinstance(exc, PromptBuildError):
            raise
        handle_failure(
            "failed to build monetisation prompt",
            exc,
            logger=logger,
        )
    data = client.ask(prompt_obj, use_memory=False, memory_manager=None, tags=full_tags)
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def send_to_evaluation_bot(event: Event, strategy: str) -> None:
    """Forward the event and strategy to the evaluation service."""
    url = os.getenv("EVAL_BOT_URL", "http://localhost:8000/evaluate")
    payload = {"event": event.__dict__, "strategy": strategy}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:  # pragma: no cover - network
        logger.warning("Failed to send data to evaluation bot")


__all__ = [
    "Event",
    "NewsDB",
    "fetch_news",
    "filter_events",
    "cluster_events",
    "ImpactPredictor",
    "monetise_event",
    "send_to_evaluation_bot",
]
