from __future__ import annotations

"""Legacy wrapper for OpenAI chat completions.

SelfCodingEngine now handles all code generation locally, removing the need
for remote payment notices. This module still exposes
:func:`chat_completion_create`, which mirrors
``openai.ChatCompletion.create`` for components that have not yet migrated.
It prepends :data:`~stripe_policy.PAYMENT_ROUTER_NOTICE` to prompts before
forwarding the request. A :class:`~prompt_types.Prompt` instance is required
and a custom ``openai_client`` can be supplied for testing.
"""

from typing import Any, Optional

from prompt_types import Prompt

from resilience import retry_with_backoff
from sandbox_settings import SandboxSettings
from .prompt_notice import prepend_payment_notice

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore


def chat_completion_create(
    prompt: Prompt,
    *,
    openai_client: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Proxy ``openai.ChatCompletion.create`` for legacy callers.

    SelfCodingEngine performs generation locally, but this helper remains for
    backward compatibility.  Callers must supply a fully built
    :class:`~prompt_types.Prompt` to ensure the context engine is exercised.
    """

    if not isinstance(prompt, Prompt):
        raise TypeError("prompt must be a Prompt")

    client = openai_client or openai
    if client is None:  # pragma: no cover - import guard
        raise RuntimeError("openai library not available")

    content = prompt.user
    if prompt.examples:
        content += "\n\n" + "\n".join(prompt.examples)

    msgs = [{"role": "user", "content": content}]
    if prompt.system:
        msgs.insert(0, {"role": "system", "content": prompt.system})

    msgs = prepend_payment_notice(msgs)

    _settings = SandboxSettings()
    delays = list(getattr(_settings, "codex_retry_delays", [2, 5, 10]))
    return retry_with_backoff(
        lambda: client.ChatCompletion.create(messages=msgs, **kwargs),
        attempts=len(delays),
        delays=delays,
    )


__all__ = ["chat_completion_create"]
