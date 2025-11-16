"""Utilities for injecting the self-coding notice into prompts.

SelfCodingEngine now handles all code generation locally, removing the need for
remote billing reminders. This module centralises usage of
:data:`PAYMENT_ROUTER_NOTICE` and exposes a helper to prepend the legacy notice
to message lists for components that still expect it.
"""

from __future__ import annotations

from typing import Dict, List

from stripe_policy import PAYMENT_ROUTER_NOTICE
try:  # pragma: no cover - optional runtime dependency
    import stripe_billing_router  # noqa: F401
except Exception:  # pragma: no cover - best effort
    stripe_billing_router = None  # type: ignore


def prepend_payment_notice(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return ``messages`` with :data:`PAYMENT_ROUTER_NOTICE` prepended.

    While SelfCodingEngine runs locally, some legacy tools still expect the
    notice to appear. If a system message already exists it is updated to
    include the notice at the beginning of its content. Otherwise a new system
    message containing the notice is inserted at the start of the list.
    """

    msgs = list(messages)
    if msgs and msgs[0].get("role") == "system":
        content = msgs[0].get("content", "")
        msgs[0]["content"] = PAYMENT_ROUTER_NOTICE + "\n" + content
    else:
        msgs = [{"role": "system", "content": PAYMENT_ROUTER_NOTICE}, *msgs]
    return msgs


__all__ = ["PAYMENT_ROUTER_NOTICE", "prepend_payment_notice"]
