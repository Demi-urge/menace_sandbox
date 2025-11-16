from __future__ import annotations

"""Shared rate limiting helpers for LLM backends.

The :class:`TokenBucket` tracks token usage per minute and blocks when the
configured allowance would be exceeded.  ``sleep_with_backoff`` implements a
simple exponential backoff strategy that can be reused by backends when
retrying requests.
"""

import threading
import time
from typing import Dict, Any

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - transformers may not be installed
    AutoTokenizer = None  # type: ignore

_ENCODER_CACHE: Dict[str, Any] = {}


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


# Singleton bucket shared by all clients.  Callers should invoke ``update_rate``
# before consuming tokens to ensure the capacity reflects the current
# configuration.
SHARED_TOKEN_BUCKET = TokenBucket()


def _get_encoder(model: str | None) -> Any | None:
    """Return a tiktoken encoder for *model* if available."""

    if not tiktoken:
        return None
    key = model or "cl100k_base"
    enc = _ENCODER_CACHE.get(key)
    if enc is not None:
        return enc
    try:
        if model:
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    _ENCODER_CACHE[key] = enc
    return enc


def _get_hf_tokenizer(model: str | None) -> Any | None:
    """Return a HuggingFace tokenizer for *model* if available."""

    if not AutoTokenizer or not model:
        return None
    key = f"hf::{model}"
    tok = _ENCODER_CACHE.get(key)
    if tok is None:
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except Exception:
            tok = None
        _ENCODER_CACHE[key] = tok
    return tok


def estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token usage for *text* using the model's tokenizer."""

    enc = _get_encoder(model)
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    tok = _get_hf_tokenizer(model)
    if tok:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    # Fallback heuristic: assume 4 characters per token
    return max(1, len(text) // 4)


def sleep_with_backoff(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> None:
    """Sleep using exponential backoff based on *attempt* number."""

    delay = min(base * (2**attempt), max_delay)
    time.sleep(delay)


__all__ = ["TokenBucket", "estimate_tokens", "sleep_with_backoff", "SHARED_TOKEN_BUCKET"]
