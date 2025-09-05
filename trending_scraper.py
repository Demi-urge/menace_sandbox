"""Trending web scraper for multiple business platforms."""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Set
import os
import asyncio
import importlib
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
import re
from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)
CACHE_FILE = Path(resolve_path(os.getenv("TREND_CACHE", "trending_cache.json")))

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

try:
    from pytrends.request import TrendReq
except Exception:  # pragma: no cover - optional dependency
    TrendReq = None  # type: ignore

try:  # optional async HTTP client
    aiohttp = importlib.import_module("aiohttp")
except Exception:  # pragma: no cover - optional dependency
    aiohttp = None  # type: ignore


@dataclass
class TrendingItem:
    """Normalized representation of a trending product or business."""

    platform: str
    niche: Optional[str] = None
    product_name: Optional[str] = None
    price_point: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    trend_signal: Optional[float] = None
    intensity: Optional[float] = None
    source_url: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


SCRAPE_METHODS = [
    "scrape_reddit",
    "scrape_shopify",
    "scrape_gumroad",
    "scrape_fiverr",
    "scrape_google_trends",
    "scrape_hackernews",
    "scrape_trendforecast",
    "scrape_social_firehose",
    "scrape_producthunt",
    "scrape_github_trending",
]

BS4_DEPENDENT = {
    "scrape_producthunt",
    "scrape_shopify",
    "scrape_gumroad",
    "scrape_fiverr",
    "scrape_github_trending",
    "scrape_hackernews",
}


class _LinkParser(HTMLParser):
    """Basic HTML anchor extractor used when BeautifulSoup is unavailable."""

    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = re.compile(pattern)
        self.results: list[tuple[str, str]] = []
        self._capture = False
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href", "")
            if self.pattern.search(href):
                self._capture = True
                self._href = href

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._capture:
            text = "".join(self._text).strip()
            self.results.append((text, self._href))
            self._capture = False
            self._href = ""
            self._text.clear()


def _parse_links(text: str, pattern: str) -> list[tuple[str, str]]:
    """Return ``(text, href)`` tuples for anchors whose href matches ``pattern``."""

    parser = _LinkParser(pattern)
    try:
        parser.feed(text)
    except Exception as exc:  # pragma: no cover - malformed HTML
        logger.warning("link parser failed: %s", exc)
    return parser.results


class BaseScraper:
    """Simple HTTP scraper with retry logic."""

    def __init__(self, retries: int = 3, delay: float = 1.0) -> None:
        if requests is not None:
            try:
                self.session = requests.Session()
            except Exception:  # pragma: no cover - misconfigured requests
                self.session = None
        else:
            self.session = None
        self._use_async = requests is None
        self._aiohttp = aiohttp
        if requests is None:
            logger.warning("requests missing; falling back to urllib%s", " with aiohttp" if aiohttp else "")
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup missing; using basic parsers")
        self.retries = retries
        self.delay = delay

    def fetch(self, url: str) -> str:
        """Return the body for ``url`` or raise ``RuntimeError`` on failure."""

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                if self.session is not None:
                    resp = self.session.get(
                        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
                    )
                    if resp.status_code == 200:
                        return resp.text
                    last_exc = RuntimeError(
                        f"non-success status {resp.status_code} from {url}"
                    )
                else:
                    if self._use_async:
                        return asyncio.run(self._fetch_async(url))
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        if resp.status == 200:
                            return resp.read().decode("utf-8", errors="ignore")
                        last_exc = RuntimeError(
                            f"non-success status {resp.status} from {url}"
                        )
            except Exception as exc:  # pragma: no cover - network issues
                last_exc = exc
                logger.warning("fetch attempt %s for %s failed: %s", attempt, url, exc)
            time.sleep(self.delay)

        raise RuntimeError(f"failed to fetch {url}") from last_exc

    async def _fetch_async(self, url: str) -> str:
        """Asynchronous fetch using ``aiohttp`` when available."""

        if self._aiohttp is not None:
            async with self._aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
                ) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    raise RuntimeError(f"non-success status {resp.status} from {url}")

        loop = asyncio.get_running_loop()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        def _load() -> str:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"non-success status {resp.status} from {url}")

        return await loop.run_in_executor(None, _load)


class TrendingScraper(BaseScraper):
    """Scraper for various trend sources."""

    def __init__(self, retries: int = 3, delay: float = 1.0) -> None:
        super().__init__(retries=retries, delay=delay)
        self.unavailable_scrapers: Set[str] = set()
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
        self._check_dependencies()

    @property
    def disabled_scrapers(self) -> Set[str]:
        """Scrapers that will not run due to missing dependencies."""
        return set(self.unavailable_scrapers)

    def _check_dependencies(self) -> None:
        """Populate :attr:`unavailable_scrapers` based on missing libraries."""
        missing: list[str] = []
        if requests is None:
            missing.append("requests")
        if BeautifulSoup is None:
            missing.append("BeautifulSoup")
        if TrendReq is None:
            missing.append("pytrends")
        if missing:
            logger.warning(
                "Missing deps %s; using fallbacks",
                ", ".join(missing),
            )

    def scrape_producthunt(self, energy: float | None = None) -> List[TrendingItem]:
        """Return top launches from ProductHunt."""
        items: List[TrendingItem] = []
        text = self.fetch("https://www.producthunt.com/feed")
        if not text:
            return items
        if BeautifulSoup:
            soup = BeautifulSoup(text, "xml")
            entries = [
                (
                    it.title.get_text(" ", strip=True),
                    it.link.get_text(strip=True),
                )
                for it in soup.find_all("item")[:10]
            ]
        else:
            try:
                root = ET.fromstring(text)
                entries = [
                    (
                        it.findtext("title", default="").strip(),
                        it.findtext("link", default="").strip(),
                    )
                    for it in root.findall(".//item")[:10]
                ]
            except Exception as exc:
                logger.warning("basic producthunt parser failed: %s", exc)
                entries = []
        for title, link in entries:
            items.append(
                TrendingItem(
                    platform="ProductHunt",
                    niche="Launch",
                    product_name=title,
                    price_point=None,
                    tags=[],
                    trend_signal=None,
                    source_url=link,
                )
            )
        return self._filter_items(items, energy)

    def scrape_github_trending(self, energy: float | None = None) -> List[TrendingItem]:
        """Return trending GitHub repositories."""
        items: List[TrendingItem] = []
        text = self.fetch("https://github.com/trending")
        if not text:
            return items
        entries: List[tuple[str, str]] = []
        if BeautifulSoup:
            soup = BeautifulSoup(text, "html.parser")
            for h in soup.select("article.Box-row h2 a")[:10]:
                name = h.get_text(" ", strip=True)
                link = h.get("href", "")
                entries.append((name, link))
        else:
            try:
                raw = _parse_links(text, r"/[^/]+/[^/]+")
                for name, link in raw[:10]:
                    entries.append((name.replace("/", " ").strip(), link))
            except Exception as exc:
                logger.warning("basic github parser failed: %s", exc)
        for name, link in entries:
            items.append(
                TrendingItem(
                    platform="GitHub",
                    niche="Repositories",
                    product_name=name,
                    price_point=None,
                    tags=[],
                    trend_signal=None,
                    source_url="https://github.com" + link,
                )
            )
        return self._filter_items(items, energy)

    @staticmethod
    def correlate_trends(
        items: List[TrendingItem], min_sources: int = 2
    ) -> List[TrendingItem]:
        """Return items appearing on at least ``min_sources`` platforms."""
        freq: dict[str, set[str]] = {}
        for it in items:
            key = (it.product_name or "").lower()
            freq.setdefault(key, set()).add(it.platform)
        correlated = [
            it
            for it in items
            if len(freq.get((it.product_name or "").lower(), set())) >= min_sources
        ]
        correlated.sort(
            key=lambda i: len(freq.get((i.product_name or "").lower(), set())),
            reverse=True,
        )
        return correlated

    def _filter_items(
        self, items: List[TrendingItem], energy: float | None
    ) -> List[TrendingItem]:
        if energy is None:
            return items
        terms = self.quick_terms if energy < 0.5 else self.scale_terms
        filtered = [
            i
            for i in items
            if any(t in (i.product_name or "").lower() for t in terms)
            or any(t in tag.lower() for tag in i.tags for t in terms)
        ]
        return filtered or items

    def scrape_reddit(self, energy: float | None = None) -> List[TrendingItem]:
        items: List[TrendingItem] = []
        url = "https://www.reddit.com/r/Entrepreneur/top.json?limit=10&t=day"
        text = self.fetch(url)
        if not text:
            return items
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return items
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            items.append(
                TrendingItem(
                    platform="Reddit",
                    niche=post.get("subreddit"),
                    product_name=post.get("title"),
                    price_point=None,
                    tags=[post.get("link_flair_text", "")],
                    trend_signal=None,
                    source_url=f"https://www.reddit.com{post.get('permalink')}",
                )
            )
        return self._filter_items(items, energy)

    def scrape_shopify(self, energy: float | None = None) -> List[TrendingItem]:
        items: List[TrendingItem] = []
        url = "https://www.shopify.com/trending"
        text = self.fetch(url)
        if not text:
            return items
        if BeautifulSoup:
            soup = BeautifulSoup(text, "html.parser")
            raw = [(c.get_text(strip=True), c.get("href")) for c in soup.select("a[href*='products']")]
        else:
            raw = _parse_links(text, r"products")
        for name, link in raw:
            items.append(
                TrendingItem(
                    platform="Shopify",
                    niche="Trending Store",
                    product_name=name,
                    price_point=None,
                    tags=[],
                    trend_signal=None,
                    source_url=link,
                )
            )
        return self._filter_items(items, energy)

    def scrape_gumroad(self, energy: float | None = None) -> List[TrendingItem]:
        items: List[TrendingItem] = []
        url = "https://gumroad.com/discover"
        text = self.fetch(url)
        if not text:
            return items
        if BeautifulSoup:
            soup = BeautifulSoup(text, "html.parser")
            cards = [
                (
                    c.select_one(".product-name").get_text(strip=True)
                    if c.select_one(".product-name")
                    else "",
                    c.select_one("a").get("href") if c.select_one("a") else "",
                    c.select_one(".price").get_text(strip=True) if c.select_one(".price") else "",
                )
                for c in soup.select("div[class*='product-card']")
            ]
        else:
            cards = [(*t, "") for t in _parse_links(text, r"/l/")]
        for name, link, price in cards:
            price_val = None
            if price:
                price_text = price.replace("$", "")
                try:
                    price_val = float("".join(c for c in price_text if (c.isdigit() or c == ".")))
                except ValueError:
                    price_val = None
            items.append(
                TrendingItem(
                    platform="Gumroad",
                    niche="Top Sellers",
                    product_name=name or None,
                    price_point=price_val,
                    tags=[],
                    trend_signal=None,
                    source_url=link or None,
                )
            )
        return self._filter_items(items, energy)

    def scrape_fiverr(self, energy: float | None = None) -> List[TrendingItem]:
        items: List[TrendingItem] = []
        url = "https://www.fiverr.com/categories/business"
        text = self.fetch(url)
        if not text:
            return items
        if BeautifulSoup:
            soup = BeautifulSoup(text, "html.parser")
            raw = [(g.get_text(strip=True), g.get("href")) for g in soup.select("a[href*='/services/']")]
        else:
            raw = _parse_links(text, r"/services/")
        for title, link in raw:
            items.append(
                TrendingItem(
                    platform="Fiverr",
                    niche="Business",
                    product_name=title,
                    price_point=None,
                    tags=[],
                    trend_signal=None,
                    source_url=f"https://www.fiverr.com{link}" if link else None,
                )
            )
        return self._filter_items(items, energy)

    def scrape_google_trends(self, energy: float | None = None) -> List[TrendingItem]:
        items: List[TrendingItem] = []
        entries: List[tuple[str, str]] = []
        if TrendReq:
            try:
                pytrends = TrendReq()
                df = pytrends.trending_searches(pn="united_states")
                entries = [(val, "https://trends.google.com") for val in df[0].tolist()]
            except Exception as exc:
                logger.warning("pytrends failed: %s", exc)
                entries = []
        else:
            text = self.fetch(
                "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
            )
            if text:
                try:
                    root = ET.fromstring(text)
                    entries = [
                        (
                            it.findtext("title", default="").strip(),
                            it.findtext("link", default="").strip(),
                        )
                        for it in root.findall(".//item")[:10]
                    ]
                except Exception as exc:
                    logger.warning("basic google trends parser failed: %s", exc)
                    entries = []
        for title, link in entries:
            items.append(
                TrendingItem(
                    platform="Google Trends",
                    niche="Rising Queries",
                    product_name=title,
                    price_point=None,
                    tags=[],
                    trend_signal=None,
                    source_url=link or "https://trends.google.com",
                )
            )
        return self._filter_items(items, energy)

    def scrape_hackernews(self, energy: float | None = None) -> List[TrendingItem]:
        items: List[TrendingItem] = []
        text = self.fetch("https://hnrss.org/frontpage")
        if not text:
            return items
        entries: List[tuple[str, str]] = []
        if BeautifulSoup:
            soup = BeautifulSoup(text, "xml")
            entries = [
                (
                    it.title.get_text(" ", strip=True),
                    it.link.get_text(strip=True),
                )
                for it in soup.find_all("item")[:10]
            ]
        else:
            try:
                root = ET.fromstring(text)
                entries = [
                    (
                        it.findtext("title", default="").strip(),
                        it.findtext("link", default="").strip(),
                    )
                    for it in root.findall(".//item")[:10]
                ]
            except Exception as exc:
                logger.warning("basic hackernews parser failed: %s", exc)
                entries = []
        for title, link in entries:
            items.append(
                TrendingItem(
                    platform="HackerNews",
                    niche="Tech",
                    product_name=title,
                    price_point=None,
                    tags=[],
                    trend_signal=None,
                    source_url=link,
                )
            )
        return self._filter_items(items, energy)

    def scrape_trendforecast(self, energy: float | None = None) -> List[TrendingItem]:
        """Collect trending items from a hypothetical TrendForecast API."""
        items: List[TrendingItem] = []
        url = "https://api.trendforecast.com/v1/trends"
        text = self.fetch(url)
        if not text:
            return items
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return items
        for entry in data.get("trends", [])[:10]:
            items.append(
                TrendingItem(
                    platform="TrendForecast",
                    niche=entry.get("category"),
                    product_name=entry.get("name"),
                    price_point=None,
                    tags=entry.get("tags", []),
                    trend_signal=float(entry.get("score", 0)),
                    source_url=entry.get("url"),
                )
            )
        return self._filter_items(items, energy)

    def scrape_social_firehose(self, energy: float | None = None) -> List[TrendingItem]:
        """Collect items from a generic social media firehose endpoint."""
        items: List[TrendingItem] = []
        url = "https://firehose.social/api/latest"
        text = self.fetch(url)
        if not text:
            return items
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return items
        for entry in data.get("items", [])[:10]:
            items.append(
                TrendingItem(
                    platform="SocialFirehose",
                    niche=entry.get("platform"),
                    product_name=entry.get("text"),
                    price_point=None,
                    tags=entry.get("tags", []),
                    trend_signal=float(entry.get("score", 0)),
                    source_url=entry.get("url"),
                )
            )
        return self._filter_items(items, energy)

    def collect_all(self, energy: float | None = None) -> List[TrendingItem]:
        methods = [getattr(self, name) for name in SCRAPE_METHODS]
        collected = []
        for m in methods:
            if m.__name__ in self.unavailable_scrapers:
                collected.append([])
                continue
            try:
                collected.append(m(energy))
            except Exception as exc:
                logger.warning("scrape %s failed: %s", m.__name__, exc)
                collected.append([])
        items = [i for sub in collected for i in sub]
        if not items and CACHE_FILE.exists():
            try:
                data = json.loads(CACHE_FILE.read_text())
                return [TrendingItem(**d) for d in data]
            except Exception as exc:
                logger.warning("failed reading cache %s: %s", CACHE_FILE, exc)
                return []
        items = self.aggregate_signals(items)
        try:
            CACHE_FILE.write_text(json.dumps([asdict(i) for i in items], indent=2))
        except Exception as exc:
            logger.warning("failed writing cache %s: %s", CACHE_FILE, exc)
        return self.correlate_trends(items)

    @staticmethod
    def aggregate_signals(items: List[TrendingItem]) -> List[TrendingItem]:
        freq: dict[str, int] = {}
        for it in items:
            key = (it.product_name or "").lower()
            freq[key] = freq.get(key, 0) + 1
        for it in items:
            it.trend_signal = float(freq.get((it.product_name or "").lower(), 1))
        return items

    @staticmethod
    def detect_microtrends(
        items: List[TrendingItem], within: int = 60 * 60 * 24, min_hits: int = 2
    ) -> List[TrendingItem]:
        """Return items trending across ``min_hits`` sources within ``within`` seconds."""
        now = time.time()
        groups: Dict[str, List[TrendingItem]] = {}
        for it in items:
            if now - it.timestamp > within:
                continue
            key = (it.product_name or "").lower()
            groups.setdefault(key, []).append(it)
        result: List[TrendingItem] = []
        for group in groups.values():
            if len({g.platform for g in group}) >= min_hits:
                intensity = len(group) / max(1.0, within / 3600)
                for g in group:
                    g.intensity = intensity
                result.extend(group)
        result.sort(key=lambda i: (i.intensity or 0.0), reverse=True)
        return result


def save_results(name: str, items: List[TrendingItem]) -> None:
    data = [asdict(i) for i in items]
    with open(f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    scraper = TrendingScraper()
    items = scraper.collect_all()
    save_results("all_trends", items)


if __name__ == "__main__":
    main()
