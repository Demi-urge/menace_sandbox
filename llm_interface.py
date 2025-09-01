from __future__ import annotations

"""Lightweight LLM abstraction with an OpenAI GPT-4o implementation.

This module exposes a minimal :class:`LLMClient` interface for language model
backends.  The interface purposely keeps the surface area tiny so it can be
re-used in small utilities without pulling in a full featured SDK.

Two concrete pieces are provided:

``Prompt``
    Simple dataclass representing the user input.

``OpenAIClient``
    Implementation talking to the OpenAI Chat Completions API using the
    GPT-4o model.  The client features basic exponential backoff retry logic
    and a per-second rate limiter.  ``generate`` returns both the raw JSON
    response as well as the parsed text content so downstream callers can
    choose the level of detail they require.
"""

from dataclasses import dataclass
from typing import Any, Dict
from abc import ABC, abstractmethod
import os
import threading
import time

import requests


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """Input shown to the language model."""

    text: str


@dataclass
class LLMResult:
    """Result of an LLM generation call."""

    raw: Dict[str, Any]
    text: str


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract base class for language model clients."""

    @abstractmethod
    def generate(self, prompt_obj: Prompt) -> LLMResult:
        """Return both raw provider JSON and parsed text for *prompt_obj*."""

    @abstractmethod
    def parse(self, raw_response: Dict[str, Any]) -> str:
        """Extract the human readable text from *raw_response*."""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple thread safe rate limiter expressed in requests per second."""

    def __init__(self, rps: float) -> None:
        self.rps = rps
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        if self.rps <= 0:
            return
        with self._lock:
            interval = 1.0 / self.rps
            now = time.time()
            wait = self._last + interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = max(now, self._last + interval)


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    """LLMClient implementation using the OpenAI GPT-4o model."""

    api_url = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        *,
        max_retries: int = 5,
        rate_limit_rps: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.max_retries = max_retries
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(rate_limit_rps)

    # ------------------------------------------------------------------
    def generate(self, prompt_obj: Prompt) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_obj.text}],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        backoff = 1.0
        for attempt in range(self.max_retries):
            self.rate_limiter.acquire()
            try:
                response = self.session.post(
                    self.api_url, json=payload, headers=headers, timeout=30
                )
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
            else:
                if response.status_code == 429 and attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if response.status_code >= 500 and attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                raw = response.json()
                text = self.parse(raw)
                return LLMResult(raw=raw, text=text)

            time.sleep(backoff)
            backoff *= 2

        raise RuntimeError("Failed to obtain completion from OpenAI")

    # ------------------------------------------------------------------
    def parse(self, raw_response: Dict[str, Any]) -> str:
        try:
            return raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""


__all__ = ["Prompt", "LLMResult", "LLMClient", "OpenAIClient"]
