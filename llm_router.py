from __future__ import annotations

"""Routing helpers for language model backends."""

from importlib import import_module
from typing import Callable, Dict

from llm_interface import Prompt, LLMResult, LLMClient
from sandbox_settings import SandboxSettings

ClientFactory = Callable[[], LLMClient]


def _factory_from_path(path: str) -> ClientFactory:
    """Resolve *path* to a callable returning an :class:`LLMClient`."""
    module_name, attr_name = path.rsplit(".", 1)
    module = import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):  # pragma: no cover - defensive
        raise TypeError(f"Backend factory at {path!r} is not callable")
    return factory


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


def client_from_settings(
    settings: SandboxSettings | None = None, *, size_threshold: int = 1000
) -> LLMClient:
    """Create an :class:`LLMClient` based on :class:`SandboxSettings`.

    ``SandboxSettings.available_backends`` maps backend names to dotted import
    paths referencing factories or classes returning :class:`LLMClient`
    instances.  The ``preferred_llm_backend`` selects the primary backend while
    ``llm_fallback_backend`` optionally specifies a secondary backend used by
    :class:`LLMRouter` as a fallback.
    """
    settings = settings or SandboxSettings()
    backends: Dict[str, str] = {
        k.lower(): v for k, v in settings.available_backends.items()
    }
    backend = settings.preferred_llm_backend or settings.llm_backend
    backend = backend.lower()
    fallback = settings.llm_fallback_backend
    fallback = fallback.lower() if fallback else None

    def _make(name: str) -> LLMClient:
        try:
            path = backends[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown LLM backend: {name}") from exc
        factory = _factory_from_path(path)
        return factory()

    primary = _make(backend)
    if not fallback:
        return primary
    fallback_client = _make(fallback)
    return LLMRouter(remote=primary, local=fallback_client, size_threshold=size_threshold)


__all__ = ["LLMRouter", "client_from_settings"]
