from __future__ import annotations


class VectorServiceError(RuntimeError):
    """Base exception for vector service failures."""


class RateLimitError(VectorServiceError):
    """Raised when the underlying service rate limits requests."""


class MalformedPromptError(ValueError):
    """Raised when input prompts are malformed or empty."""


__all__ = [
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]
