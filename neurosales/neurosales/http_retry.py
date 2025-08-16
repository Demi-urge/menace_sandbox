from __future__ import annotations

import random
import time

import logging
import requests

logger = logging.getLogger(__name__)


class ResilientRequester:
    """HTTP wrapper with backoff and jitter for GET/POST."""

    def __init__(self, max_retries: int = 3, base_backoff: float = 0.5) -> None:
        self.session = requests.Session()
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    # ------------------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        attempt = 0
        while True:
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code >= 400:
                    raise requests.HTTPError(f"status {resp.status_code}")
                return resp
            except Exception as e:
                logger.exception("HTTP %s request failed", method)
                attempt += 1
                if self.max_retries and attempt > self.max_retries:
                    raise RuntimeError(
                        f"Request to {url} failed after {attempt} attempts"
                    ) from e
                delay = self.base_backoff * (2 ** (attempt - 1))
                jitter = random.random()
                time.sleep(delay * (1 + jitter))

    # ------------------------------------------------------------------
    def get(self, url: str, **kwargs) -> requests.Response:
        return self._request("GET", url, **kwargs)

    # ------------------------------------------------------------------
    def post(self, url: str, **kwargs) -> requests.Response:
        return self._request("POST", url, **kwargs)
