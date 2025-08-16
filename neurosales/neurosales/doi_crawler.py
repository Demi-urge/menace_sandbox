from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from .static_harvest import hardened_session

logger = logging.getLogger(__name__)

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


@dataclass
class DOICrawlRecord:
    doi: str
    source_url: str
    fetch_depth: int


class DOICrawler:
    """Crawl academic sites for DOIs across paginated results and reference trees."""

    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self.session = hardened_session()
        self.seen: Set[str] = set()
        self.records: List[DOICrawlRecord] = []

    # ------------------------------------------------------------------
    def _extract_dois(self, html: str) -> List[str]:
        return DOI_RE.findall(html)

    # ------------------------------------------------------------------
    def _find_next_link(self, html: str, base_url: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        next_a = soup.find("a", string=lambda s: s and "next" in s.lower())
        if next_a and next_a.get("href"):
            return urljoin(base_url, next_a["href"])
        return None

    # ------------------------------------------------------------------
    def _crawl_references(self, doi: str, depth: int) -> None:
        if depth >= self.max_depth:
            return
        url = f"https://doi.org/{doi}"
        try:
            resp = self.session.get(url, timeout=10)
            html = resp.text
        except Exception as e:
            logger.exception("Failed to fetch DOI page %s", url)
            raise RuntimeError(f"Failed to fetch {url}") from e
        for sub_doi in self._extract_dois(html):
            if sub_doi not in self.seen:
                self.seen.add(sub_doi)
                self.records.append(
                    DOICrawlRecord(doi=sub_doi, source_url=url, fetch_depth=depth)
                )
                self._crawl_references(sub_doi, depth + 1)

    # ------------------------------------------------------------------
    def _crawl_search_page(self, url: str, depth: int) -> None:
        try:
            resp = self.session.get(url, timeout=10)
            html = resp.text
        except Exception as e:
            logger.exception("Failed to fetch search page %s", url)
            raise RuntimeError(f"Failed to fetch {url}") from e

        new_doi_found = False
        for doi in self._extract_dois(html):
            if doi not in self.seen:
                new_doi_found = True
                self.seen.add(doi)
                self.records.append(
                    DOICrawlRecord(doi=doi, source_url=url, fetch_depth=depth)
                )
                self._crawl_references(doi, depth + 1)

        next_url = self._find_next_link(html, url)
        if new_doi_found and next_url:
            self._crawl_search_page(next_url, depth + 1)

    # ------------------------------------------------------------------
    def crawl(self, start_urls: List[str], out_csv: str) -> None:
        for url in start_urls:
            self._crawl_search_page(url, 0)

        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["doi", "source_url", "fetch_depth"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

