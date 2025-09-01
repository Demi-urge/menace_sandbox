"""Minimal OpenAI client implementing the LLMClient protocol."""

from __future__ import annotations

from typing import Dict, Any

import requests

from llm_interface import Prompt, LLMResult, LLMClient

try:  # pragma: no cover - package vs module import
    from . import llm_config, rate_limit
except Exception:  # pragma: no cover - stand-alone usage
    import llm_config  # type: ignore
    import rate_limit  # type: ignore


class OpenAILLMClient(LLMClient):
    """Simple client for the OpenAI chat completions API."""

    api_url = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        max_retries: int | None = None,
    ) -> None:
        cfg = llm_config.get_config()
        model = model or cfg.model
        super().__init__(model)
        self.api_key = api_key or cfg.api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.max_retries = max_retries or cfg.max_retries
        self._session = requests.Session()
        self._rate_limiter = rate_limit.TokenBucket(cfg.tokens_per_minute)

    # ------------------------------------------------------------------
    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST *payload* to the OpenAI API with retry/backoff."""

        cfg = llm_config.get_config()
        self._rate_limiter.update_rate(cfg.tokens_per_minute)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        retries = cfg.max_retries
        for attempt in range(retries):
            tokens = rate_limit.estimate_tokens(
                " ".join(m.get("content", "") for m in payload.get("messages", [])),
                model=self.model,
            )
            self._rate_limiter.consume(tokens)
            try:
                response = self._session.post(
                    self.api_url, headers=headers, json=payload, timeout=30
                )
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
            else:
                if response.status_code == 429 and attempt < retries - 1:
                    rate_limit.sleep_with_backoff(attempt)
                    continue
                response.raise_for_status()
                return response.json()
            rate_limit.sleep_with_backoff(attempt)

        raise RuntimeError("Exceeded maximum retries for OpenAI request")

    # ------------------------------------------------------------------
    def _generate(self, prompt: Prompt) -> LLMResult:
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        for ex in prompt.examples:
            messages.append({"role": "system", "content": ex})
        messages.append({"role": "user", "content": prompt.user})

        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if prompt.tags:
            payload["tags"] = prompt.tags
        if prompt.vector_confidence is not None:
            payload["vector_confidence"] = prompt.vector_confidence

        raw = self._request(payload)
        text = ""
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        return LLMResult(raw=raw, text=text)
