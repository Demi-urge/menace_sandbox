"""Text Research Bot for extracting and summarising textual data."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Callable
import os
import logging
from collections import Counter
import re

try:
    from nltk.tokenize import sent_tokenize  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sent_tokenize = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore

logger = logging.getLogger(__name__)

from .db_router import DBRouter

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

try:
    import PyPDF2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PyPDF2 = None  # type: ignore

try:
    from gensim.summarization import summarize  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    summarize = None  # type: ignore


@dataclass
class TextSource:
    """Container for extracted text with metadata."""

    content: str
    url: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


def _download(session: requests.Session, url: str) -> str:
    try:
        resp = session.get(url, timeout=10)
    except Exception:
        return ""
    return resp.text if resp.status_code == 200 else ""


def _parse_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


def _parse_pdf(path: Path) -> str:
    if not PyPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(str(path))
    except Exception:
        return ""
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                texts.append(t)
        except Exception:
            continue
    return " ".join(texts)


def summarise_text(text: str, ratio: float = 0.2) -> str:
    text = text.strip()
    if not text:
        return ""

    if summarize:
        try:
            result = summarize(text, ratio=ratio)
            if result:
                return result
        except Exception:
            logger.exception("summarize() failed")

    if sent_tokenize:
        try:
            sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            sentences = []
    else:
        sentences = []
    if not sentences:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if len(sentences) <= 1:
        return text

    if TfidfVectorizer:
        try:
            vec = TfidfVectorizer()
            matrix = vec.fit_transform(sentences)
            scores = matrix.sum(axis=1).A1
            ranked = [s for s, _ in sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)]
            count = max(1, int(len(ranked) * ratio))
            return ". ".join(ranked[:count]).rstrip(".") + "."
        except Exception:
            logger.exception("TF-IDF summarization failed")

    # frequency fallback if sklearn unavailable
    words = re.findall(r"\w+", text.lower())
    freq = Counter(words)
    ranked = sorted(
        sentences,
        key=lambda s: sum(freq.get(w.lower(), 0) for w in s.split()),
        reverse=True,
    )
    count = max(1, int(len(ranked) * ratio))
    return ". ".join(ranked[:count]).rstrip(".") + "."


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class TextResearchBot:
    """Scrape, parse and summarise textual sources."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        send_callback: Optional[Callable[[Iterable[TextSource]], None]] = None,
        db_router: Optional[DBRouter] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.send_callback = send_callback
        self.db_router = db_router

    def extract_from_urls(self, urls: Iterable[str]) -> List[TextSource]:
        sources: List[TextSource] = []
        for url in urls:
            html = _download(self.session, url)
            if not html:
                continue
            text = _parse_html(html)
            if len(text.split()) < 5:
                continue
            sources.append(TextSource(content=text, url=url))
        return sources

    def extract_from_files(self, paths: Iterable[Path]) -> List[TextSource]:
        sources: List[TextSource] = []
        for path in paths:
            text = _parse_pdf(path)
            if not text or len(text.split()) < 5:
                continue
            sources.append(TextSource(content=text, metadata={"file": str(path)}))
        return sources

    def summarise_sources(self, sources: Iterable[TextSource], ratio: float = 0.2) -> List[TextSource]:
        results: List[TextSource] = []
        for src in sources:
            summary = summarise_text(src.content, ratio=ratio)
            if summary:
                results.append(TextSource(content=summary, url=src.url, metadata=src.metadata))
        return results

    def process(self, urls: Iterable[str], files: Iterable[Path], ratio: float = 0.2) -> List[TextSource]:
        if self.db_router:
            try:
                hits = self.db_router.query_all(" ".join(urls)).info
            except Exception:
                hits = []
            if hits:
                return [
                    TextSource(content=h.summary or h.content, url=h.source_url)
                    for h in hits
                ]
        web = self.extract_from_urls(urls)
        docs = self.extract_from_files(files)
        combined = web + docs
        summaries = self.summarise_sources(combined, ratio=ratio)
        if self.send_callback:
            self.send_callback(summaries)
        else:
            send_to_aggregator(summaries)
        return summaries


def send_to_aggregator(items: Iterable[TextSource]) -> None:
    """POST ``items`` to the Research Aggregator service."""

    if not requests:
        return
    url = os.getenv("AGGREGATOR_URL", "http://localhost:8000/aggregate")
    payload = [item.__dict__ for item in items]
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:  # pragma: no cover - network issues
        logger.warning("Failed to send data to aggregator")


__all__ = ["TextSource", "TextResearchBot", "summarise_text", "send_to_aggregator"]