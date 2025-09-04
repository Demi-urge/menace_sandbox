"""Utilities for injecting the payment router notice into prompts.

This module centralises usage of :data:`PAYMENT_ROUTER_NOTICE` and exposes a
small helper to prepend the notice to OpenAI style message lists.  The helper
ensures that all outbound prompts remind the model that billing is routed
through the approved payment router.
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

    If a system message already exists it is updated to include the notice at
    the beginning of its content.  Otherwise a new system message containing the
    notice is inserted at the start of the list.
    """

    msgs = list(messages)
    if msgs and msgs[0].get("role") == "system":
        content = msgs[0].get("content", "")
        msgs[0]["content"] = PAYMENT_ROUTER_NOTICE + "\n" + content
    else:
        msgs = [{"role": "system", "content": PAYMENT_ROUTER_NOTICE}, *msgs]
    return msgs


__all__ = ["PAYMENT_ROUTER_NOTICE", "prepend_payment_notice"]
