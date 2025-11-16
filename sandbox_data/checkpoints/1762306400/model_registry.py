"""Mapping of model backend names to :class:`LLMClient` implementations."""

from __future__ import annotations

from typing import Dict, Type

from llm_interface import LLMClient, OpenAIProvider

try:  # pragma: no cover - optional dependencies
    from local_client import OllamaClient, VLLMClient
except Exception:  # pragma: no cover - if module missing
    OllamaClient = VLLMClient = None  # type: ignore[assignment]


MODEL_REGISTRY: Dict[str, Type[LLMClient]] = {}
MODEL_REGISTRY["openai"] = OpenAIProvider
if OllamaClient is not None:
    MODEL_REGISTRY["ollama"] = OllamaClient
if VLLMClient is not None:
    MODEL_REGISTRY["vllm"] = VLLMClient


def get_client(name: str, **kwargs) -> LLMClient:
    """Instantiate an :class:`LLMClient` for *name*.

    Parameters
    ----------
    name:
        Key in :data:`MODEL_REGISTRY` identifying the backend.
    **kwargs:
        Additional keyword arguments forwarded to the client constructor.
    """

    try:
        cls = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown model backend: {name}") from exc
    return cls(**kwargs)


__all__ = ["MODEL_REGISTRY", "get_client"]
