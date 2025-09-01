from __future__ import annotations

"""Backend loading local model weights via HuggingFace transformers."""

from dataclasses import dataclass, field
from typing import Any
import time

import llm_config
import rate_limit
from llm_interface import Prompt, LLMResult, LLMBackend, LLMClient

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover - transformers may not be installed
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore


@dataclass
class LocalWeightsBackend(LLMBackend):
    """Minimal backend running a model from local weights."""

    model: str
    device: str = "cpu"
    _tokenizer: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _rate_limiter: rate_limit.TokenBucket = field(init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - simple initialiser
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required for LocalWeightsBackend")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(self.model)
        if torch is not None:
            self._model.to(self.device)
        cfg = llm_config.get_config()
        self._rate_limiter = rate_limit.SHARED_TOKEN_BUCKET
        self._rate_limiter.update_rate(getattr(cfg, "tokens_per_minute", 0))

    def generate(self, prompt: Prompt) -> LLMResult:
        cfg = llm_config.get_config()
        retries = cfg.max_retries
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)

        for attempt in range(retries):
            self._rate_limiter.update_rate(cfg.tokens_per_minute)
            self._rate_limiter.consume(prompt_tokens)
            try:
                inputs = self._tokenizer(prompt.text, return_tensors="pt")
                if torch is not None:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                start = time.perf_counter()
                output_ids = self._model.generate(**inputs, max_new_tokens=256)
                latency_ms = (time.perf_counter() - start) * 1000
                gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
                text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
                completion_tokens = rate_limit.estimate_tokens(
                    text, model=self.model
                )
                self._rate_limiter.consume(completion_tokens)
                return LLMResult(
                    raw={"backend": "local_weights", "model": self.model},
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )
            except Exception:  # pragma: no cover - generation failure
                if attempt == retries - 1:
                    raise
            rate_limit.sleep_with_backoff(attempt)

        raise RuntimeError("Failed to obtain completion from local weights backend")


def local_weights_client(model: str | None = None, device: str = "cpu") -> LLMClient:
    """Return an :class:`LLMClient` using :class:`LocalWeightsBackend`."""

    model = model or llm_config.get_config().model
    backend = LocalWeightsBackend(model=model, device=device)
    return LLMClient(model=backend.model, backends=[backend], log_prompts=False)


__all__ = ["LocalWeightsBackend", "local_weights_client"]
