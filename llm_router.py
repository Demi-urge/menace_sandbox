from __future__ import annotations

"""Routing helpers for language model backends."""

from typing import Dict, Callable

from llm_interface import Prompt, LLMResult, LLMClient
from sandbox_settings import SandboxSettings

# Deferred imports to avoid pulling in heavy dependencies on module import.
ClientFactory = Callable[[], LLMClient]


def _openai_factory() -> LLMClient:  # pragma: no cover - simple wrapper
    from openai_client import OpenAILLMClient
    return OpenAILLMClient()


def _ollama_factory() -> LLMClient:  # pragma: no cover - simple wrapper
    from local_client import OllamaClient
    return OllamaClient()


def _vllm_factory() -> LLMClient:  # pragma: no cover - simple wrapper
    from local_client import VLLMClient
    return VLLMClient()


_FACTORIES: Dict[str, ClientFactory] = {
    "openai": _openai_factory,
    "ollama": _ollama_factory,
    "vllm": _vllm_factory,
}


class LLMRouter(LLMClient):
    """Route requests between remote and local backends.

    The router chooses the remote backend for larger prompts while favouring
    the local backend for smaller ones.  If the selected backend raises an
    exception, the other backend is attempted as a fallback.
    """

    def __init__(self, remote: LLMClient, local: LLMClient, *, size_threshold: int = 1000) -> None:
        self.remote = remote
        self.local = local
        self.size_threshold = size_threshold

    def generate(self, prompt: Prompt) -> LLMResult:
        primary = self.remote if len(prompt.text) > self.size_threshold else self.local
        fallback = self.local if primary is self.remote else self.remote
        try:
            return primary.generate(prompt)
        except Exception:
            return fallback.generate(prompt)


def client_from_settings(settings: SandboxSettings | None = None, *, size_threshold: int = 1000) -> LLMClient:
    """Create an :class:`LLMClient` based on :class:`SandboxSettings`.

    When a fallback backend is configured, an :class:`LLMRouter` is returned
    that dispatches between the primary (assumed remote) and fallback
    (assumed local) backends.
    """
    settings = settings or SandboxSettings()
    backend = settings.llm_backend.lower()
    fallback = settings.llm_fallback_backend
    fallback = fallback.lower() if fallback else None

    try:
        primary_factory = _FACTORIES[backend]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown LLM backend: {backend}") from exc
    primary = primary_factory()

    if not fallback:
        return primary

    try:
        fallback_factory = _FACTORIES[fallback]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown fallback backend: {fallback}") from exc
    fallback_client = fallback_factory()
    return LLMRouter(remote=primary, local=fallback_client, size_threshold=size_threshold)


__all__ = ["LLMRouter", "client_from_settings"]
