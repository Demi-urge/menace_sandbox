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

from llm_interface import Prompt, LLMResult, LLMClient


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
        super().__init__(model=model, base_url=base_url)

    def generate(self, prompt: Prompt) -> LLMResult:
        payload = {"model": self.model, "prompt": prompt.text}
        raw = self._post("api/generate", payload)
        text = raw.get("response", "") or raw.get("text", "")
        return LLMResult(raw=raw, text=text)


class VLLMClient(_BaseLocalClient, LLMClient):
    """Client for a vLLM REST server."""

    def __init__(self, model: str = "facebook/opt-125m", base_url: str = "http://localhost:8000") -> None:
        super().__init__(model=model, base_url=base_url)

    def generate(self, prompt: Prompt) -> LLMResult:
        payload = {"model": self.model, "prompt": prompt.text}
        raw = self._post("generate", payload)
        text = raw.get("text") or raw.get("generated_text", "")
        return LLMResult(raw=raw, text=text)


__all__ = ["OllamaClient", "VLLMClient"]
