"""Lightweight LLM interface definitions.

The project only needs a very small slice of functionality from whatever
language model backend is in use.  To keep the dependency surface minimal this
module defines tiny dataclasses for exchanging prompts and results together with
a base :class:`LLMClient` that backends can inherit from.  The class exposes a
single :py:meth:`generate` method returning an :class:`LLMResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import os
import time
import json
import requests


@dataclass(slots=True)
class Prompt:
    """Input to an LLM generation call.

    ``text`` contains the assembled prompt shown to the model while ``examples``
    stores any illustrative snippets that were used to build the prompt.
    Additional information such as ROI metrics or tone preferences can be
    attached via the ``metadata`` mapping.
    """

    text: str
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    # The original codebase often treated prompts as raw strings.  To ease the
    # transition to the structured :class:`Prompt` object, the dataclass mimics
    # ``str`` behaviour for common operations.  This allows existing callers
    # that perform string operations such as ``in`` checks or ``.index`` calls to
    # continue working without modification while still exposing the structured
    # fields.

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text

    def __eq__(self, other: object) -> bool:  # pragma: no cover - defensive
        if isinstance(other, Prompt):
            return (
                self.text == other.text
                and self.examples == other.examples
                and self.metadata == other.metadata
            )
        if isinstance(other, str):
            return self.text == other
        return False

    def __contains__(self, item: str) -> bool:  # pragma: no cover - delegation
        return item in self.text

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.text, name)

    def __add__(self, other: object):  # pragma: no cover - delegation
        if isinstance(other, Prompt):
            return self.text + other.text
        if isinstance(other, str):
            return self.text + other
        return NotImplemented

    def __radd__(self, other: object):  # pragma: no cover - delegation
        if isinstance(other, str):
            return other + self.text
        if isinstance(other, Prompt):
            return other.text + self.text
        return NotImplemented


@dataclass(slots=True)
class LLMResult:
    """Result returned by an :class:`LLMClient`.

    ``text`` holds the raw string produced by the model.  ``parsed`` optionally
    stores a structured representation of that string (for example JSON
    decoded from ``text``).  ``raw`` can be used by clients to stash any
    transport specific payload such as HTTP responses.  ``completions``
    exposes the raw text of all choices returned by the model.
    """

    text: str = ""
    parsed: Any | None = None
    raw: Dict[str, object] = field(default_factory=dict)
    completions: List[str] = field(default_factory=list)


class LLMClient(ABC):
    """Base class describing the minimal LLM client interface."""

    @abstractmethod
    def generate(self, prompt: Prompt) -> LLMResult:  # pragma: no cover - interface
        """Generate a response for *prompt*.

        Implementations should return an :class:`LLMResult` with the model's
        response in ``text`` and may optionally populate ``parsed`` and
        ``raw`` with backend-specific data.
        """


class OpenAIProvider(LLMClient):
    """Minimal OpenAI chat completion client using GPT-4o."""

    api_url = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self.model = model or "gpt-4o"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    # ------------------------------------------------------------------
    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST *payload* to the OpenAI API with retries/backoff."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    self.api_url, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
            else:
                if response.status_code == 429:
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()
                else:
                    if response.ok:
                        return response.json()
                    if attempt == self.max_retries - 1:
                        response.raise_for_status()

            time.sleep(backoff)
            backoff *= 2

        raise RuntimeError("Exceeded maximum retries for OpenAI request")

    # ------------------------------------------------------------------
    def generate(self, prompt: Prompt) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.text}],
        }

        raw = self._request(payload)
        completions: List[str] = []
        text = ""
        try:
            for choice in raw.get("choices", []):
                message = choice.get("message", {})
                content = message.get("content", "")
                completions.append(content)
            if completions:
                text = completions[0]
        except Exception:  # pragma: no cover - defensive
            pass

        parsed = None
        if text:
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None

        return LLMResult(text=text, parsed=parsed, raw=raw, completions=completions)


class OllamaProvider(LLMClient):
    """Client for an `ollama` local model server."""

    base_url = "http://localhost:11434"

    def __init__(self, model: str = "mixtral", base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url or self.base_url
        self._session = requests.Session()

    @classmethod
    def is_available(cls, base_url: str | None = None) -> bool:
        url = (base_url or cls.base_url).rstrip("/") + "/api/tags"
        try:
            requests.get(url, timeout=1)
        except requests.RequestException:
            return False
        return True

    def generate(self, prompt: Prompt) -> LLMResult:
        payload = {"model": self.model, "prompt": prompt.text}
        url = self.base_url.rstrip("/") + "/api/generate"
        response = self._session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        raw = response.json()
        text = raw.get("response", "") or raw.get("text", "")
        return LLMResult(raw=raw, text=text)


class VLLMProvider(LLMClient):
    """Client for a vLLM HTTP server."""

    base_url = "http://localhost:8000"

    def __init__(self, model: str = "llama3", base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url or self.base_url
        self._session = requests.Session()

    @classmethod
    def is_available(cls, base_url: str | None = None) -> bool:
        try:
            requests.get(base_url or cls.base_url, timeout=1)
        except requests.RequestException:
            return False
        return True

    def generate(self, prompt: Prompt) -> LLMResult:
        payload = {"model": self.model, "prompt": prompt.text}
        url = self.base_url.rstrip("/") + "/generate"
        response = self._session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        raw = response.json()
        text = raw.get("text") or raw.get("generated_text", "")
        return LLMResult(raw=raw, text=text)


class HybridProvider(LLMClient):
    """Automatically route requests between local and remote providers."""

    def __init__(self, *, size_threshold: int = 1000) -> None:
        self.size_threshold = size_threshold

        local: LLMClient | None = None
        if OllamaProvider.is_available():
            local = OllamaProvider()
        elif VLLMProvider.is_available():
            local = VLLMProvider()
        self.local = local

        remote: LLMClient | None = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                remote = OpenAIProvider()
            except Exception:
                remote = None
        self.remote = remote

        if not self.local and not self.remote:
            raise RuntimeError("No available LLM providers")

    def generate(self, prompt: Prompt) -> LLMResult:
        use_local = self.local is not None and (
            self.remote is None or len(prompt.text) <= self.size_threshold
        )
        if use_local:
            try:
                return self.local.generate(prompt)
            except Exception:
                if self.remote:
                    return self.remote.generate(prompt)
                raise
        if self.remote:
            try:
                return self.remote.generate(prompt)
            except Exception:
                if self.local:
                    return self.local.generate(prompt)
                raise
        assert self.local  # for type checking
        return self.local.generate(prompt)


__all__ = [
    "Prompt",
    "LLMResult",
    "LLMClient",
    "OpenAIProvider",
    "OllamaProvider",
    "VLLMProvider",
    "HybridProvider",
]
