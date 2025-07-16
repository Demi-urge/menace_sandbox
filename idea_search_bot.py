"""Idea Search Bot for discovering potential business models online.

This module provides utilities to generate search queries, submit them to the
Google Custom Search API and analyse the returned results. A simple NLP based
scoring is applied to determine relevance. Found ideas can be handed off to the
``DatabaseManagementBot`` for further processing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Dict

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:
    import nltk
    from nltk.tokenize import word_tokenize
    try:
        word_tokenize("test")
    except LookupError:
        nltk = None  # type: ignore
        word_tokenize = lambda text: text.split()  # type: ignore
except Exception:  # nltk may not be installed during tests
    nltk = None  # type: ignore
    word_tokenize = lambda text: text.split()  # type: ignore

from .database_management_bot import DatabaseManagementBot


@dataclass
class KeywordBank:
    """Collection of topics and phrases used for query generation."""

    topics: List[str] = field(default_factory=lambda: [
        "ai",
        "sustainability",
        "automation",
        "remote work",
        "crypto",
        "fintech",
        "healthtech",
        "education",
    ])
    phrases: List[str] = field(default_factory=lambda: [
        "how to start",
        "online business",
        "new model",
        "startup idea",
    ])
    quick_topics: List[str] = field(
        default_factory=lambda: [
            "freelancing",
            "affiliate marketing",
            "dropshipping",
            "digital products",
        ]
    )
    scale_topics: List[str] = field(
        default_factory=lambda: [
            "enterprise platform",
            "infrastructure",
            "saas",
            "marketplace",
        ]
    )
    quick_phrases: List[str] = field(
        default_factory=lambda: [
            "quick profit",
            "low investment",
            "fast roi",
            "low startup cost",
        ]
    )
    scale_phrases: List[str] = field(
        default_factory=lambda: [
            "long term growth",
            "scalable model",
            "market domination",
            "scalable platform",
        ]
    )

    def generate_queries(self, energy: float) -> List[str]:
        """Combine topics and phrases into distinct search queries."""
        topics = list(self.topics)
        phrases = list(self.phrases)
        if energy < 0.5:
            topics = self.quick_topics + topics
            phrases = self.quick_phrases + phrases
        else:
            topics = self.scale_topics + topics
            phrases = self.scale_phrases + phrases

        queries = []
        for topic in topics:
            for phrase in phrases:
                queries.append(f"{phrase} {topic}")
        return queries


@dataclass
class GoogleSearchClient:
    """Thin wrapper around the Google Custom Search API with rate limiting."""

    api_key: str
    engine_id: str
    session: requests.Session | None = None
    backoff: float = 1.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.session:
            self.session = requests.Session()

    def search(self, query: str) -> Dict[str, object]:
        """Submit a query to the Google Custom Search API with retries."""
        params = {"key": self.api_key, "cx": self.engine_id, "q": query}
        delay = self.backoff
        for _ in range(self.max_retries):
            try:
                resp = self.session.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                )
            except Exception:
                resp = None
            if resp and resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    return {}
            if resp and resp.status_code not in {429, 500, 503}:
                break
            time.sleep(delay)
            delay *= 2
        return {}


@dataclass
class Result:
    title: str
    link: str
    snippet: str
    score: float = 0.0


def extract_results(data: Dict[str, object]) -> List[Result]:
    """Convert API JSON data into ``Result`` objects."""
    items = data.get("items", []) if isinstance(data, dict) else []
    results: List[Result] = []
    for item in items:
        results.append(
            Result(
                title=str(item.get("title", "")),
                link=str(item.get("link", "")),
                snippet=str(item.get("snippet", "")),
            )
        )
    return results


def score_text(text: str, keywords: Iterable[str]) -> float:
    """Simple keyword frequency scoring using nltk when available."""
    if nltk:
        tokens = [w.lower() for w in word_tokenize(text)]
    else:
        tokens = text.lower().split()
    return sum(tokens.count(k.lower()) for k in keywords)


def rank_results(
    results: List[Result],
    keywords: Iterable[str],
    bias_phrases: Iterable[str] | None = None,
) -> List[Result]:
    """Score and sort results by keyword relevance with optional bias."""
    bias_phrases = list(bias_phrases or [])
    for r in results:
        text = r.title + " " + r.snippet
        base = score_text(text, keywords)
        bias = sum(1 for p in bias_phrases if p.lower() in text.lower())
        r.score = base + bias * 2
    results.sort(key=lambda r: r.score, reverse=True)
    seen = set()
    unique: List[Result] = []
    for r in results:
        if r.link not in seen and r.score > 0:
            unique.append(r)
            seen.add(r.link)
    return unique


def cross_check(results: List[Result]) -> List[Result]:
    """Legacy hook for result filtering."""
    return results


def discover_new_models(
    client: GoogleSearchClient,
    bank: KeywordBank,
    *,
    energy: float,
) -> List[Result]:
    """High level convenience function running the full search pipeline."""
    discovered: List[Result] = []
    queries = bank.generate_queries(energy)
    bias = bank.quick_phrases if energy < 0.5 else bank.scale_phrases
    for q in queries:
        data = client.search(q)
        if not data:
            continue
        results = extract_results(data)
        ranked = rank_results(results, q.split(), bias)
        new = cross_check(ranked)
        discovered.extend(new)
    return discovered


def handoff_to_database(
    result: Result,
    tags: Iterable[str],
    *,
    db_bot: DatabaseManagementBot | None = None,
    source: str = "idea_search_bot",
) -> None:
    """Send a discovered idea to the database management bot."""
    bot = db_bot or DatabaseManagementBot()
    # ``ingest_idea`` returns a status but we intentionally ignore it to
    # keep communication one-way.
    bot.ingest_idea(result.title, tags=tags, source=source, urls=[result.link])

