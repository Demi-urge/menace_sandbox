from __future__ import annotations

"""Lightweight LLM abstraction with an OpenAI GPT-4o implementation.

This module exposes a minimal :class:`LLMClient` interface for language model
backends.  The interface purposely keeps the surface area tiny so it can be
re-used in small utilities without pulling in a full featured SDK.

Two concrete pieces are provided:

``Prompt``
    Dataclass describing the different pieces of a chat style prompt.  It is
    defined in :mod:`prompt_types` but re-exported here for convenience.

``OpenAIClient``
    Implementation talking to the OpenAI Chat Completions API using the
    GPT-4o model.  The client features basic exponential backoff retry logic
    and a per-second rate limiter.  ``generate`` returns both the raw JSON
    response as well as the parsed text content so downstream callers can
    choose the level of detail they require.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol
from abc import ABC, abstractmethod
import os
import threading
import time

import requests


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

# ``Prompt`` lives in a separate module so other packages can import it
# without pulling in the entire client implementation.
try:  # pragma: no cover - package vs module import
    from .prompt_types import Prompt
except Exception:  # pragma: no cover - stand-alone usage
    from prompt_types import Prompt


@dataclass
class LLMResult:
    """Result of an LLM generation call."""

    raw: Dict[str, Any]
    text: str
    parsed: Any | None = None


# Backwards compatibility -------------------------------------------------
# ``Completion`` previously named the ``LLMResult`` container.  Keep an alias
# so older imports continue to work without modification.
Completion = LLMResult


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
# Backend protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Minimal protocol for lightweight local backends."""

    model: str

    def generate(self, prompt: Prompt) -> LLMResult:  # pragma: no cover - interface
        ...


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
        messages: List[Dict[str, str]] = []
        if prompt_obj.system:
            messages.append({"role": "system", "content": prompt_obj.system})
        for ex in prompt_obj.examples:
            messages.append({"role": "system", "content": ex})
        messages.append({"role": "user", "content": prompt_obj.user})

        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if prompt_obj.tags:
            payload["tags"] = prompt_obj.tags
        if prompt_obj.vector_confidence is not None:
            payload["vector_confidence"] = prompt_obj.vector_confidence

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
                result = LLMResult(raw=raw, text=text)
                try:
                    from prompt_db import log_interaction

                    log_interaction(prompt_obj, raw, text, prompt_obj.tags)
                except Exception:
                    pass  # Logging should never break generation
                return result

            time.sleep(backoff)
            backoff *= 2

        raise RuntimeError("Failed to obtain completion from OpenAI")

    # ------------------------------------------------------------------
    def parse(self, raw_response: Dict[str, Any]) -> str:
        try:
            return raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""


__all__ = [
    "Prompt",
    "LLMResult",
    "Completion",
    "LLMClient",
    "LLMBackend",
    "OpenAIClient",
]
