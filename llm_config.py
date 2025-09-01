from __future__ import annotations

"""Configuration helpers for language model backends.

This module centralises reading of model selection, API keys, retry limits
and rate limiting thresholds.  Environment variables take precedence over
values specified in an optional JSON configuration file referenced via the
``LLM_CONFIG_FILE`` environment variable.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import json
import os


@dataclass
class LLMConfig:
    """Simple container holding configuration for LLM backends."""

    model: str = "gpt-4o"
    api_key: str | None = None
    max_retries: int = 5
    tokens_per_minute: int = 0
    pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _load_file(path: str | None) -> Dict[str, Any]:
    """Load configuration from *path* if it exists, else return an empty dict."""

    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def get_config() -> LLMConfig:
    """Return the current configuration for LLM backends."""

    file_cfg = _load_file(os.getenv("LLM_CONFIG_FILE"))
    model = os.getenv("LLM_MODEL") or file_cfg.get("model") or "gpt-4o"
    api_key = os.getenv("OPENAI_API_KEY") or file_cfg.get("api_key")
    max_retries = int(
        os.getenv("LLM_MAX_RETRIES", file_cfg.get("max_retries", 5))
    )
    tokens_per_minute = int(
        os.getenv("LLM_TPM", file_cfg.get("tokens_per_minute", 0))
    )
    pricing: Dict[str, Dict[str, float]] = {}
    file_pricing = file_cfg.get("pricing")
    if isinstance(file_pricing, dict):
        pricing.update(file_pricing)
    env_pricing = os.getenv("LLM_PRICING")
    if env_pricing:
        try:
            pricing.update(json.loads(env_pricing))
        except Exception:
            pass
    return LLMConfig(
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        tokens_per_minute=tokens_per_minute,
        pricing=pricing,
    )


__all__ = ["LLMConfig", "get_config"]
