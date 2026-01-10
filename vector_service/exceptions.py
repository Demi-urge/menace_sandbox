from __future__ import annotations


class VectorServiceError(RuntimeError):
    """Base exception for vector service failures."""


class RateLimitError(VectorServiceError):
    """Raised when the underlying service rate limits requests."""


class MalformedPromptError(VectorServiceError, ValueError):
    """Raised when input prompts are malformed or empty."""


class RetrieverConfigurationError(VectorServiceError):
    """Raised when retriever configuration is missing required DB inputs."""


__all__ = [
    "VectorServiceError",
    "RetrieverConfigurationError",
    "RateLimitError",
    "MalformedPromptError",
]
