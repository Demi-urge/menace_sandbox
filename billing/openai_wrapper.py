from __future__ import annotations

"""Lightweight OpenAI wrapper that injects the payment notice.

This module exposes :func:`chat_completion_create` which mirrors
``openai.ChatCompletion.create`` but automatically prepends
:data:`~stripe_policy.PAYMENT_ROUTER_NOTICE` to the ``messages`` list and
appends compressed retrieval context.  The wrapper requires a
``ContextBuilder`` instance and will raise a descriptive error when one is
not supplied.  A custom ``openai_client`` can be provided for easy testing.
"""

import json
from typing import Any, Dict, List, Optional

from vector_service.context_builder import ContextBuilder
from snippet_compressor import compress_snippets

from resilience import retry_with_backoff
from sandbox_settings import SandboxSettings
from .prompt_notice import prepend_payment_notice

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


def chat_completion_create(
    messages: List[Dict[str, str]],
    *,
    openai_client: Optional[Any] = None,
    context_builder: ContextBuilder,
    **kwargs: Any,
) -> Any:
    """Proxy ``openai.ChatCompletion.create`` with payment notice injection."""

    if context_builder is None:
        raise TypeError("context_builder is required for chat_completion_create")

    client = openai_client or openai
    if client is None:  # pragma: no cover - import guard
        raise RuntimeError("openai library not available")

    msgs = prepend_payment_notice(messages)

    # Build retrieval context from the latest user message and compress it
    query = messages[-1]["content"] if messages else ""
    ctx_res = context_builder.build(query)
    ctx = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
    if isinstance(ctx, (dict, list)):
        ctx = json.dumps(ctx, separators=(",", ":"))
    ctx = compress_snippets({"snippet": ctx}).get("snippet", ctx)
    msgs.append({"role": "system", "content": ctx})

    _settings = SandboxSettings()
    delays = list(getattr(_settings, "codex_retry_delays", [2, 5, 10]))
    return retry_with_backoff(
        lambda: client.ChatCompletion.create(messages=msgs, **kwargs),
        attempts=len(delays),
        delays=delays,
    )


__all__ = ["chat_completion_create"]
