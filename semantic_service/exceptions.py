from __future__ import annotations


class SemanticServiceError(RuntimeError):
    """Base class for errors raised by :mod:`semantic_service`."""


class VectorServiceError(SemanticServiceError):
    """Base exception for vector service failures."""


class RateLimitError(VectorServiceError):
    """Raised when the underlying service rate limits requests."""


class MalformedPromptError(SemanticServiceError, ValueError):
    """Raised when input prompts are malformed or empty."""


__all__ = [
    "SemanticServiceError",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]
