from __future__ import annotations

"""Anthropic API client implementing the LLMBackend protocol."""

from typing import Any, Dict, List
import os
import time
import requests

import rate_limit
import llm_config

from llm_interface import Prompt, LLMResult, LLMClient


class AnthropicClient(LLMClient):
    """Lightweight client for the Anthropic Messages API."""

    api_url = "https://api.anthropic.com/v1/messages"

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
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or cfg.api_key
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required")
        self.max_retries = max_retries or cfg.max_retries
        self._rate_limiter = rate_limit.TokenBucket(cfg.tokens_per_minute)

    # ------------------------------------------------------------------
    def _prepare_payload(self, prompt: Prompt) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        for ex in getattr(prompt, "examples", []):
            messages.append({"role": "system", "content": ex})
        messages.append({"role": "user", "content": prompt.user})
        return {"model": self.model, "max_tokens": 1024, "messages": messages}

    # ------------------------------------------------------------------
    def _generate(self, prompt: Prompt) -> LLMResult:
        payload = self._prepare_payload(prompt)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        cfg = llm_config.get_config()
        retries = self.max_retries
        prompt_tokens_est = rate_limit.estimate_tokens(
            " ".join(m.get("content", "") for m in payload["messages"]),
            model=self.model,
        )
        for attempt in range(retries):
            self._rate_limiter.update_rate(cfg.tokens_per_minute)
            self._rate_limiter.consume(prompt_tokens_est)
            try:
                start = time.perf_counter()
                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=30
                )
                latency_ms = (time.perf_counter() - start) * 1000
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
            else:
                if response.status_code == 429 and attempt < retries - 1:
                    rate_limit.sleep_with_backoff(attempt)
                    continue
                if response.status_code >= 500 and attempt < retries - 1:
                    rate_limit.sleep_with_backoff(attempt)
                    continue
                response.raise_for_status()
                raw = response.json()
                text = ""
                try:
                    parts = raw.get("content", [])
                    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                except Exception:
                    pass
                usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
                prompt_tokens = usage.get("input_tokens") or prompt_tokens_est
                completion_tokens = usage.get("output_tokens") or rate_limit.estimate_tokens(
                    text, model=self.model
                )
                extra = max(
                    0, (prompt_tokens or 0) + (completion_tokens or 0) - prompt_tokens_est
                )
                if extra:
                    self._rate_limiter.consume(extra)
                return LLMResult(
                    raw=raw,
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )
        raise RuntimeError("Anthropic request failed")


__all__ = ["AnthropicClient"]
