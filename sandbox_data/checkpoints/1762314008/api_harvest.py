from __future__ import annotations

import random
import subprocess
import os
from typing import List, Optional

import logging
import requests

logger = logging.getLogger(__name__)


class APIScraper:
    """Fetch neuroscience data from official APIs and proxy services."""

    def __init__(self, proxies: Optional[List[str]] = None) -> None:
        self.session = requests.Session()
        if proxies is None:
            env = os.getenv("NEURO_PROXY_LIST", "")
            proxies = [p.strip() for p in env.split(",") if p.strip()] if env else None
        self.proxies = proxies or []

    # ------------------------------------------------------------------
    def _next_proxy(self) -> Optional[str]:
        if not self.proxies:
            return None
        return random.choice(self.proxies)

    # ------------------------------------------------------------------
    def fetch_pubmed_xml(self, pmids: List[str]) -> str:
        """Return PubMed XML via NCBI E-utilities."""
        ids = ",".join(pmids)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        resp = self.session.get(
            url, params={"db": "pubmed", "id": ids, "retmode": "xml"}, timeout=10
        )
        return resp.text

    # ------------------------------------------------------------------
    def fetch_citation_context(self, paper_id: str) -> dict:
        """Retrieve citation context from Semantic Scholar GraphQL."""
        query = (
            "query($id: ID!) { paper(id: $id) { title citationCount } }"
        )
        resp = self.session.post(
            "https://api.semanticscholar.org/graphql",
            json={"query": query, "variables": {"id": paper_id}},
            timeout=10,
        )
        return resp.json()

    # ------------------------------------------------------------------
    def fetch_crossref_metadata(self, doi: str) -> dict:
        """Get publisher metadata from CrossRef REST API."""
        url = f"https://api.crossref.org/works/{doi}"
        resp = self.session.get(url, timeout=10)
        return resp.json()

    # ------------------------------------------------------------------
    def download_kaggle_dataset(self, dataset: str, dest: str) -> bool:
        """Download an EEG/FNIRS dataset via the Kaggle CLI."""
        try:
            subprocess.check_call(
                ["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "-q"]
            )
            return True
        except Exception as e:
            logger.exception("Failed to download Kaggle dataset %s", dataset)
            raise RuntimeError("Failed to download Kaggle dataset") from e

    # ------------------------------------------------------------------
    def fetch_fmri_atlas_json(self, url: str) -> dict:
        """Use a scraping proxy to return rendered JSON for pay-walled atlases."""
        proxy = self._next_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None
        resp = self.session.get(url, proxies=proxies, timeout=10)
        return resp.json()
