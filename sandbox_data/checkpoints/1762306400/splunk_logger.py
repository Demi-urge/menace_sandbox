from __future__ import annotations
"""Minimal Splunk HTTP Event Collector wrapper."""

from typing import Any, Dict
import logging

from .retry_utils import retry

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


class SplunkHEC:
    """Send events to Splunk using the HTTP Event Collector."""

    def __init__(self, url: str = "http://localhost:8088", token: str = "", index: str = "main") -> None:
        self.url = url.rstrip("/") + "/services/collector/event"
        self.token = token
        self.index = index
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, ts: str, body: Dict[str, Any]) -> None:
        if requests is None:
            return
        headers = {"Authorization": f"Splunk {self.token}"}
        data = {"time": ts, "event": body, "index": self.index}

        @retry(Exception, attempts=3)
        def _post() -> object:
            return requests.post(self.url, headers=headers, json=data, timeout=1)

        try:
            _post()
        except Exception as exc:
            self.logger.error("failed to send event to Splunk: %s", exc)


__all__ = ["SplunkHEC"]
