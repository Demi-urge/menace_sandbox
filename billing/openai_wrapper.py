from __future__ import annotations

"""Legacy wrapper for OpenAI chat completions.

SelfCodingEngine now handles all code generation locally, removing the need
for remote payment notices. This module still exposes
:func:`chat_completion_create`, which mirrors
``openai.ChatCompletion.create`` for components that have not yet migrated.
It prepends :data:`~stripe_policy.PAYMENT_ROUTER_NOTICE` to the ``messages``
list and expands the latest user query via ``ContextBuilder.build_prompt``
before forwarding the request. A ``ContextBuilder`` instance is required and a
custom ``openai_client`` can be supplied for testing.
"""

from typing import Any, Dict, List, Optional

from vector_service.context_builder import ContextBuilder

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

    top_k = int(kwargs.pop("top_k", 5) or 5)
    intent = kwargs.pop("intent", None)

    query = ""
    last_user = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user = i
            query = messages[i].get("content", "")
            break

    updated_msgs = list(messages)
    if query.strip():
        prompt = context_builder.build_prompt(query, top_k=top_k, intent=intent)
        content = prompt.user
        if prompt.examples:
            content += "\n\n" + "\n".join(prompt.examples)
        if last_user is not None:
            updated_msgs[last_user]["content"] = content
        else:
            updated_msgs.append({"role": "user", "content": content})
        if prompt.system:
            updated_msgs.insert(0, {"role": "system", "content": prompt.system})

    msgs = prepend_payment_notice(updated_msgs)

    _settings = SandboxSettings()
    delays = list(getattr(_settings, "codex_retry_delays", [2, 5, 10]))
    return retry_with_backoff(
        lambda: client.ChatCompletion.create(messages=msgs, **kwargs),
        attempts=len(delays),
        delays=delays,
    )


__all__ = ["chat_completion_create"]
