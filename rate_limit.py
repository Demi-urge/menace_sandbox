from __future__ import annotations

"""Shared rate limiting helpers for LLM backends.

The :class:`TokenBucket` tracks token usage per minute and blocks when the
configured allowance would be exceeded.  ``sleep_with_backoff`` implements a
simple exponential backoff strategy that can be reused by backends when
retrying requests.
"""

import threading
import time
from typing import Optional


class TokenBucket:
    """Token based rate limiter allowing *tokens_per_minute* usage."""

    def __init__(self, tokens_per_minute: int = 0) -> None:
        self.capacity = tokens_per_minute
        self.tokens = tokens_per_minute
        self.reset_time = time.time() + 60
        self._lock = threading.Lock()

    def update_rate(self, tokens_per_minute: int) -> None:
        """Adjust the bucket capacity to *tokens_per_minute*."""

        with self._lock:
            self.capacity = tokens_per_minute
            if self.tokens > tokens_per_minute:
                self.tokens = tokens_per_minute

    def consume(self, tokens: int) -> None:
        """Consume *tokens*, blocking if allowance is exceeded."""

        if self.capacity <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                if now >= self.reset_time:
                    self.tokens = self.capacity
                    self.reset_time = now + 60
                if tokens <= self.tokens:
                    self.tokens -= tokens
                    return
                wait = self.reset_time - now
            time.sleep(wait)


def estimate_tokens(text: str) -> int:
    """Very small heuristic to estimate token usage from *text*."""

    # Rough heuristic: assume 4 characters per token
    return max(1, len(text) // 4)


def sleep_with_backoff(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> None:
    """Sleep using exponential backoff based on *attempt* number."""

    delay = min(base * (2**attempt), max_delay)
    time.sleep(delay)


__all__ = ["TokenBucket", "estimate_tokens", "sleep_with_backoff"]
