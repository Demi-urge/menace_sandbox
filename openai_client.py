
"""Deprecated OpenAI client wrapper.

This module retains backwards compatibility by re-exporting
:class:`~llm_interface.OpenAIProvider` under the old name
``OpenAILLMClient``.  Prefer importing :class:`OpenAIProvider` from
``llm_interface`` directly.
"""

from __future__ import annotations

import warnings

from llm_interface import OpenAIProvider as OpenAILLMClient, OpenAIProvider

warnings.warn(
    "openai_client.OpenAILLMClient is deprecated; use llm_interface.OpenAIProvider",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OpenAIProvider", "OpenAILLMClient"]
