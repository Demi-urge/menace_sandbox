from __future__ import annotations

import csv
import random
from dataclasses import dataclass, asdict
from typing import List

import requests
from bs4 import BeautifulSoup


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]


def hardened_session() -> requests.Session:
    """Return a session with rotated user agent and ephemeral cookie."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    sess.cookies.set("sessionid", str(random.random()))
    return sess


@dataclass
class HarvestRecord:
    url: str
    title: str = ""
    methods: str = ""
    roi: str = ""
    dopamine_graphs: int = 0
    buy_button: str = ""


class StaticHarvester:
    """Pulls and parses static HTML sources for neuroscience sales signals."""

    def _parse_dom(self, html: str) -> HarvestRecord:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        h1 = soup.find("h1")
        if not title and h1:
            title = h1.get_text(strip=True)

        methods = ""
        heading = soup.find(
            ["h1", "h2", "h3", "h4", "h5"], string=lambda s: s and "method" in s.lower()
        )
        if heading:
            parts: List[str] = []
            for sib in heading.find_all_next():
                if sib.name and sib.name.startswith("h") and sib != heading:
                    break
                parts.append(sib.get_text(" ", strip=True))
            methods = " ".join(parts)

        roi = ""
        table = soup.find(
            "table", string=lambda s: s and "roi" in s.lower() or "region" in s.lower()
        )
        if not table:
            table = soup.find("table", {"class": lambda c: c and "roi" in c.lower()})
        if table:
            roi = table.get_text(" ", strip=True)

        dopamine_graphs = len(
            soup.find_all(
                "img",
                src=lambda s: s and "dopamine" in s.lower(),
            )
        )
        dopamine_graphs += len(
            soup.find_all("img", alt=lambda s: s and "dopamine" in s.lower())
        )

        buy_button = ""
        metric = soup.find(string=lambda s: s and "buy-button" in s.lower())
        if metric:
            buy_button = metric.strip()

        return HarvestRecord(
            url="",
            title=title,
            methods=methods,
            roi=roi,
            dopamine_graphs=dopamine_graphs,
            buy_button=buy_button,
        )

    # ------------------------------------------------------------------
    def crawl(self, urls: List[str], out_csv: str) -> None:
        """Fetch URLs and write harvested data to a CSV file."""
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "url",
                "title",
                "methods",
                "roi",
                "dopamine_graphs",
                "buy_button",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for url in urls:
                session = hardened_session()
                try:
                    resp = session.get(url, timeout=10)
                    record = self._parse_dom(resp.text)
                    record.url = url
                    writer.writerow(asdict(record))
                except Exception:
                    writer.writerow({"url": url})


