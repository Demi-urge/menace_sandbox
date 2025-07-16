"""Research Fallback Bot for external intelligence gathering."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional
import re
from collections import Counter
import math

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

try:  # pragma: no cover - optional
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - optional dependency
    async_playwright = None  # type: ignore

try:  # pragma: no cover - optional
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from .research_aggregator_bot import InfoDB, ResearchItem
from .chatgpt_research_bot import summarise_text
from .proxy_manager import get_proxy


@dataclass
class FallbackResult:
    url: str
    summary: str
    embedding: Optional[List[float]]


class ResearchFallbackBot:
    """Fallback bot using headless Chrome to gather external insights."""

    def __init__(
        self,
        info_db: Optional[InfoDB] = None,
        *,
        max_retries: int = 3,
        backoff: float = 1.0,
        embedding_dim: int = 64,
    ) -> None:
        self.info_db = info_db or InfoDB()
        self.max_retries = max_retries
        self.backoff = backoff
        if SentenceTransformer:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:  # pragma: no cover - runtime download issues
                self.embedder = None
        else:
            self.embedder = None
        self.embedding_dim = embedding_dim

    async def _browse(self, url: str, proxy: Optional[str]) -> str:
        if not async_playwright:
            return ""
        for attempt in range(self.max_retries):
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context(proxy={"server": proxy} if proxy else None)
                    page = await context.new_page()
                    await page.goto(url)
                    content = await page.content()
                    await browser.close()
                    return content
            except Exception:
                await asyncio.sleep(self.backoff * (2 ** attempt))
                proxy = get_proxy()
        return ""

    def fetch_page(self, url: str) -> str:
        proxy = get_proxy()
        if async_playwright:
            return asyncio.run(self._browse(url, proxy))
        try:
            import requests  # type: ignore

            resp = requests.get(
                url,
                proxies={"http": proxy, "https": proxy} if proxy else None,
                timeout=10,
            )
            return resp.text if resp.status_code == 200 else ""
        except Exception:
            return ""

    def _summarise(self, html: str) -> str:
        if BeautifulSoup:
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        else:  # simple fallback if bs4 is unavailable
            text = re.sub(r"<[^>]+>", " ", html)
            text = " ".join(text.split())
        return summarise_text(text, ratio=0.2)

    def _embed(self, text: str) -> Optional[List[float]]:
        if self.embedder:
            try:
                return self.embedder.encode([text])[0].tolist()
            except Exception:  # pragma: no cover - runtime issues
                pass
        # simple hashed tf-idf fallback
        tokens = re.findall(r"\w+", text.lower())
        dim = self.embedding_dim
        if not tokens:
            return [0.0] * dim
        counts = Counter(tokens)
        n = len(tokens)
        vec = [0.0] * dim
        for word, c in counts.items():
            idx = hash(word) % dim
            tf = c / n
            idf = math.log((n + 1) / (c + 1)) + 1
            vec[idx] += tf * idf
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    def process(self, query: str) -> List[FallbackResult]:
        sources = [
            f"https://stackoverflow.com/search?q={query}",
            f"https://github.com/search?q={query}&type=issues",
            f"https://docs.python.org/3/search.html?q={query}",
            f"https://www.google.com/search?q={query}",
        ]
        results: List[FallbackResult] = []
        for url in sources:
            html = self.fetch_page(url)
            if not html:
                continue
            summary = self._summarise(html)
            embedding = self._embed(summary)
            item = ResearchItem(
                topic=query,
                content=summary,
                timestamp=time.time(),
                title=query,
                tags=["fallback"],
                source_url=url,
                summary=summary,
            )
            self.info_db.add(item, embedding=embedding)
            results.append(FallbackResult(url=url, summary=summary, embedding=embedding))
        return results


__all__ = ["ResearchFallbackBot", "FallbackResult"]
