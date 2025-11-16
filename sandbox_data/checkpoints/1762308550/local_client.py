from __future__ import annotations

"""Local LLM client implementations.

This module provides lightweight wrappers around local language model
servers so they can be used through the :class:`llm_interface.LLMClient`
protocol.  The goal is to offer drop-in replacements for the remote
clients used elsewhere in the codebase.
"""

from dataclasses import dataclass
from typing import Dict, Any

import requests
import rate_limit

from llm_interface import Prompt, LLMResult, LLMClient
try:  # pragma: no cover - optional during tests
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - allow stub
    class ContextBuilder:  # type: ignore
        pass


@dataclass
class _BaseLocalClient:
    """Shared logic for simple HTTP based local LLM servers."""

    model: str
    base_url: str

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()


class OllamaClient(_BaseLocalClient, LLMClient):
    """Client for the `ollama` local LLM server."""

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434") -> None:
        LLMClient.__init__(self, model)
        _BaseLocalClient.__init__(self, model=model, base_url=base_url)

    def _generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> LLMResult:
        payload = {"model": self.model, "prompt": prompt.text}
        raw = self._post("api/generate", payload)
        text = raw.get("response", "") or raw.get("text", "")
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)
        completion_tokens = rate_limit.estimate_tokens(text, model=self.model)
        return LLMResult(
            raw=raw,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class VLLMClient(_BaseLocalClient, LLMClient):
    """Client for a vLLM REST server."""

    def __init__(
        self,
        model: str = "facebook/opt-125m",
        base_url: str = "http://localhost:8000",
    ) -> None:
        LLMClient.__init__(self, model)
        _BaseLocalClient.__init__(self, model=model, base_url=base_url)

    def _generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> LLMResult:
        payload = {"model": self.model, "prompt": prompt.text}
        raw = self._post("generate", payload)
        text = raw.get("text") or raw.get("generated_text", "")
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)
        completion_tokens = rate_limit.estimate_tokens(text, model=self.model)
        return LLMResult(
            raw=raw,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = ["OllamaClient", "VLLMClient"]
