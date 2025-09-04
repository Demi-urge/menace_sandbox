from __future__ import annotations

"""Lightweight OpenAI wrapper that injects the payment notice.

This module exposes :func:`chat_completion_create` which mirrors
``openai.ChatCompletion.create`` but automatically prepends
:data:`~stripe_policy.PAYMENT_ROUTER_NOTICE` to the ``messages`` list.
It allows passing a custom ``openai_client`` for easy testing.
"""

from typing import Any, Dict, List, Optional

from .prompt_notice import prepend_payment_notice

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


def chat_completion_create(
    messages: List[Dict[str, str]],
    *,
    openai_client: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Proxy ``openai.ChatCompletion.create`` with payment notice injection."""

    client = openai_client or openai
    if client is None:  # pragma: no cover - import guard
        raise RuntimeError("openai library not available")
    msgs = prepend_payment_notice(messages)
    return client.ChatCompletion.create(messages=msgs, **kwargs)


__all__ = ["chat_completion_create"]
