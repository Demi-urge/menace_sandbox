from __future__ import annotations

"""Utilities to merge micro-model signals into Codex prompts.

The :func:`inject_prefix` helper optionally prefixes either a raw prompt string
or a list of ChatCompletion-style messages with additional context produced by a
micro-model.  The injection only occurs when enabled via configuration and when
``confidence`` is above the configured threshold.

Configuration sources (in order of precedence):

* Environment variables ``PREFIX_INJECTION_ENABLED`` and
  ``PREFIX_INJECTION_THRESHOLD``.
* Optional YAML file pointed to by ``PREFIX_INJECTION_CONFIG`` containing
  ``enabled`` and ``threshold`` keys.

This lightweight module avoids importing heavy dependencies so it can be used
by any component that prepares prompts for the SelfCodingEngine's local APIs.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

try:  # pragma: no cover - yaml is an optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

PromptType = Union[str, List[Dict[str, str]]]


def _load_config() -> tuple[bool, float]:
    """Return ``(enabled, threshold)`` based on env vars or optional YAML."""

    enabled = os.getenv("PREFIX_INJECTION_ENABLED", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    try:
        threshold = float(os.getenv("PREFIX_INJECTION_THRESHOLD", "0.8"))
    except ValueError:
        threshold = 0.8

    cfg_path = os.getenv("PREFIX_INJECTION_CONFIG")
    if cfg_path and yaml is not None:
        path = Path(cfg_path)
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text()) or {}
                if isinstance(data, dict):
                    enabled = bool(data.get("enabled", enabled))
                    th = data.get("threshold")
                    if th is not None:
                        try:
                            threshold = float(th)
                        except ValueError:
                            pass
            except Exception:
                logger.warning(
                    "Failed to load prefix injection config from %s",
                    path,
                    exc_info=True,
                )
    return enabled, threshold


def inject_prefix(
    prompt: PromptType,
    prefix: str,
    confidence: float,
    *,
    role: str = "system",
) -> PromptType:
    """Merge ``prefix`` into ``prompt`` when confidence is high.

    ``prompt`` may be either a plain string or a list of ``{"role": ..., "content": ...}``
    dictionaries compatible with ``openai.ChatCompletion`` messages.  When a
    list is supplied, the message with matching ``role`` is prefixed or, if not
    present, a new message is inserted at the start.
    """

    enabled, threshold = _load_config()
    if not enabled or confidence < threshold or not prefix.strip():
        return prompt

    if isinstance(prompt, str):
        return prefix.strip() + "\n\n" + prompt

    for msg in prompt:
        if msg.get("role") == role:
            msg["content"] = prefix.strip() + "\n\n" + msg.get("content", "")
            break
    else:
        prompt.insert(0, {"role": role, "content": prefix.strip()})
    return prompt


__all__ = ["inject_prefix"]
