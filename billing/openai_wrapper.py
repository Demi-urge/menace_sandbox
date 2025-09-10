from __future__ import annotations

"""Legacy wrapper for OpenAI chat completions.

SelfCodingEngine now handles all code generation locally, removing the need
for remote payment notices. This module still exposes
:func:`chat_completion_create`, which mirrors
``openai.ChatCompletion.create`` for components that have not yet migrated.
It prepends :data:`~stripe_policy.PAYMENT_ROUTER_NOTICE` to the ``messages``
list and appends compressed retrieval context. A ``ContextBuilder``
instance is required and a custom ``openai_client`` can be supplied for
testing.
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
    """Proxy ``openai.ChatCompletion.create`` for legacy callers.

    SelfCodingEngine performs generation locally, but this helper remains for
    backward compatibility.
    """

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
