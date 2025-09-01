"""Minimal OpenAI client implementing the LLMClient protocol."""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, Any

import requests

from llm_interface import Prompt, LLMResult, LLMClient


class OpenAILLMClient(LLMClient):
    """Simple client for the OpenAI chat completions API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(model or "gpt-4o")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self._session = requests.Session()
        # Rate limit per second; default 1 if not provided
        self._rps = float(os.getenv("OPENAI_RATE_LIMIT_RPS", "1"))
        self._lock = threading.Lock()
        self._last_request = 0.0

    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Sleep to respect the configured requests-per-second rate."""

        if self._rps <= 0:
            return
        min_interval = 1.0 / self._rps
        with self._lock:
            now = time.time()
            wait = self._last_request + min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()

    # ------------------------------------------------------------------
    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST *payload* to the OpenAI API with retry/backoff."""

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        backoff = 1.0
        for attempt in range(5):
            self._throttle()
            try:
                response = self._session.post(url, headers=headers, json=payload, timeout=30)
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code == 429 and attempt < 4:
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            return response.json()

        raise RuntimeError("Exceeded maximum retries for OpenAI request")

    # ------------------------------------------------------------------
    def _generate(self, prompt: Prompt) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.text}],
        }

        raw = self._request(payload)
        text = ""
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        return LLMResult(raw=raw, text=text)
