"""Passive Discovery Bot for passively monitoring idea sources."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional
import os
import logging
import time

from dynamic_path_router import resolve_path

from .resilience import (
    CircuitBreaker,
    CircuitOpenError,
    PublishError,
    ResilienceError,
    retry_with_backoff,
)
from .logging_utils import set_correlation_id

registry = BotRegistry()
data_bot = DataBot(start_server=False)

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore

from .prediction_manager_bot import PredictionManager

from . import database_manager

KEYWORD_DB = resolve_path("keyword_bank.json")


@dataclass
class ContentItem:
    """Structured representation of a discovered idea source."""

    text: str
    source: str
    url: str = ""
    engagement: int = 0
    tags: List[str] = field(default_factory=list)
    sentiment: float = 0.0
    novelty: float = 0.0
    monetisation: str = ""
    industry: str = ""


class DiscoveryError(ResilienceError):
    """Raised when passive discovery fails permanently."""


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class PassiveDiscoveryBot:
    """Passive network of crawlers fetching idea related content."""

    prediction_profile = {"scope": ["discovery"], "risk": ["low"]}

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        keyword_db: Path = KEYWORD_DB,
        *,
        prediction_manager: "PredictionManager" | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.keyword_db = keyword_db
        self.keywords = self._load_keywords()
        if spacy:
            print("[DEBUG] Current PATH during spacy load:", os.environ["PATH"])
            self._nlp_model = spacy.load("en_core_web_sm")
        else:
            self._nlp_model = None
        self.prediction_manager = prediction_manager
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.quick_terms = [
            "quick profit",
            "fast roi",
            "low investment",
        ]
        self.scale_terms = [
            "platform",
            "scalable",
            "infrastructure",
        ]
        self._circuit = CircuitBreaker()

    def _apply_prediction_bots(self, base: float, feats: Iterable[float]) -> float:
        """Combine predictions from assigned bots."""
        if not self.prediction_manager:
            return base
        score = base
        count = 1
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                val = pred(list(feats))
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception:
                continue
        return float(score / count)

    def _load_keywords(self) -> List[str]:
        if self.keyword_db.exists():
            try:
                return json.loads(self.keyword_db.read_text())
            except Exception:
                return []
        return []

    def _save_keywords(self, words: Iterable[str]) -> None:
        data = sorted(set(self.keywords).union(words))
        self.keyword_db.write_text(json.dumps(data, indent=2))
        self.keywords = data

    async def _fetch(self, url: str, params: Optional[dict[str, str]] = None) -> str:
        """Fetch ``url`` using retry and circuit breaker."""

        def _do() -> str:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.text

        try:
            return await asyncio.to_thread(
                lambda: retry_with_backoff(
                    lambda: self._circuit.call(_do), logger=logger
                )
            )
        except CircuitOpenError as exc:
            logger.error("fetch circuit open for %s: %s", url, exc)
            raise DiscoveryError("circuit open") from exc
        except Exception as exc:
            logger.error("fetch failed for %s", url, exc_info=True)
            return ""

    async def _fetch_reddit(self) -> List[ContentItem]:
        text = await self._fetch("https://www.reddit.com/r/Entrepreneur/top.json")
        if not text:
            return []
        try:
            data = json.loads(text)
        except Exception:
            return []
        items = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            items.append(
                ContentItem(
                    text=(d.get("title", "") + " " + d.get("selftext", "")).strip(),
                    source="reddit",
                    url="https://reddit.com" + str(d.get("permalink", "")),
                    engagement=int(d.get("score", 0)),
                )
            )
        return items

    async def _fetch_twitter(self) -> List[ContentItem]:
        token = os.getenv("TWITTER_BEARER_TOKEN")
        if not token:
            return []
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {"query": "startup", "max_results": "10", "tweet.fields": "public_metrics"}
        try:
            resp = await asyncio.to_thread(
                self.session.get,
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
        except Exception:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        items = []
        for tw in data.get("data", []):
            metrics = tw.get("public_metrics", {})
            items.append(
                ContentItem(
                    text=str(tw.get("text", "")),
                    source="twitter",
                    url=f"https://twitter.com/i/web/status/{tw.get('id')}",
                    engagement=int(metrics.get("like_count", 0)),
                )
            )
        return items

    async def _fetch_substack(self) -> List[ContentItem]:
        text = await self._fetch("https://r.jina.ai/https://substack.com/discover")
        if not text:
            return []
        soup = BeautifulSoup(text, "html.parser")
        items = []
        for a in soup.select("a[href*='/p/']")[:10]:
            title = a.get_text(" ", strip=True)
            link = a.get("href", "")
            if not link.startswith("http"):
                link = "https://substack.com" + link
            if title:
                items.append(ContentItem(text=title, source="substack", url=link))
        return items

    async def _fetch_quora(self) -> List[ContentItem]:
        text = await self._fetch("https://r.jina.ai/https://www.quora.com")
        if not text:
            return []
        soup = BeautifulSoup(text, "html.parser")
        items = []
        for h in soup.find_all("h2"):
            link = h.find("a")
            if not link:
                continue
            title = link.get_text(strip=True)
            url = link.get("href", "")
            if not url.startswith("http"):
                url = "https://www.quora.com" + url
            items.append(ContentItem(text=title, source="quora", url=url))
            if len(items) >= 10:
                break
        return items

    async def _fetch_indie_hackers(self) -> List[ContentItem]:
        text = await self._fetch("https://r.jina.ai/https://www.indiehackers.com")
        if not text:
            return []
        soup = BeautifulSoup(text, "html.parser")
        items = []
        for post in soup.select("h3"):
            link = post.find("a")
            if not link:
                continue
            title = link.get_text(strip=True)
            url = link.get("href", "")
            if not url.startswith("http"):
                url = "https://www.indiehackers.com" + url
            items.append(ContentItem(text=title, source="indiehackers", url=url))
            if len(items) >= 10:
                break
        return items

    async def _fetch_blogs(self) -> List[ContentItem]:
        text = await self._fetch("https://hnrss.org/frontpage")
        if not text:
            return []
        soup = BeautifulSoup(text, "xml")
        items = []
        for item in soup.find_all("item")[:10]:
            title = item.title.get_text(" ", strip=True)
            link = item.link.get_text(strip=True)
            items.append(ContentItem(text=title, source="blog", url=link))
        return items

    def _tokenise(self, text: str) -> List[str]:
        if self._nlp_model:
            doc = self._nlp_model(text)
            return [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
        soup = BeautifulSoup(text, "html.parser")
        words = [w.lower() for w in soup.get_text(" ").split() if w.isalpha()]
        return words

    def _novelty(self, words: Iterable[str], engagement: int) -> float:
        new_words = sum(1 for w in words if w not in self.keywords)
        duplication_penalty = 0.0
        if database_manager.search_models(" ".join(words[:3])):
            duplication_penalty = 2.0
        base = new_words + engagement / 100 - duplication_penalty
        return self._apply_prediction_bots(base, [new_words, engagement])

    def _process(self, item: ContentItem) -> ContentItem:
        words = self._tokenise(item.text)
        item.tags = words[:5]
        item.novelty = self._novelty(words, item.engagement)
        self._save_keywords(item.tags)
        return item

    async def collect(self, *, energy: float | None = None) -> List[ContentItem]:
        """Gather and process items from various sources.

        If ``energy`` is supplied, results are biased toward either short term
        or scalable opportunities.  Values below ``0.5`` favour items containing
        quick-return phrases while higher values prefer long term or scalable
        concepts.
        """
        cid = f"discovery-{int(time.time() * 1000)}"
        set_correlation_id(cid)
        try:
            tasks = [
                self._fetch_reddit(),
                self._fetch_twitter(),
                self._fetch_substack(),
                self._fetch_quora(),
                self._fetch_indie_hackers(),
                self._fetch_blogs(),
            ]
            results = await asyncio.gather(*tasks)
            items = [i for sub in results for i in sub]
            processed = [self._process(i) for i in items]
            if energy is not None:
                terms = self.quick_terms if energy < 0.5 else self.scale_terms
                filtered = [
                    i for i in processed if any(t in i.text.lower() for t in terms)
                ]
                if filtered:
                    processed = filtered
            return processed
        finally:
            set_correlation_id(None)

    async def run_cycle(self, delay: int = 3600) -> None:
        while True:
            items = await self.collect()
            for it in items:
                send_to_stage2(it)
            await asyncio.sleep(delay)


def send_to_stage2(item: ContentItem) -> None:
    """Forward a discovered item to Stage 2 via HTTP."""

    try:
        import requests  # type: ignore
    except Exception:  # pragma: no cover - optional
        return
    url = os.getenv("STAGE2_URL", "http://localhost:8000/stage2")

    circuit = send_to_stage2._circuit  # type: ignore[attr-defined]

    def _post() -> object:
        return requests.post(url, json=item.__dict__, timeout=5)

    cid = f"stage2-{int(time.time() * 1000)}"
    set_correlation_id(cid)
    try:
        retry_with_backoff(lambda: circuit.call(_post), logger=logging.getLogger(__name__))
    except CircuitOpenError as exc:
        logging.getLogger(__name__).error("Stage 2 circuit open: %s", exc)
        raise PublishError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network
        logging.getLogger(__name__).warning("Failed to forward item to Stage 2: %s", exc)
    finally:
        set_correlation_id(None)


send_to_stage2._circuit = CircuitBreaker()  # type: ignore[attr-defined]


__all__ = ["ContentItem", "PassiveDiscoveryBot", "send_to_stage2"]
