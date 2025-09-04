"""Compatibility wrappers around :mod:`stripe_billing_router`.

This module exists for backwards compatibility.  New code should import and use
``stripe_billing_router`` directly, which provides routing, key management and
strict error handling.  The thin wrappers defined here simply delegate to the
router and surface its results.
"""

from __future__ import annotations

from typing import Any

from .stripe_billing_router import get_balance as _router_get_balance, init_charge


def is_enabled() -> bool:
    """Stripe support is always enabled.

    All configuration and error handling is performed by
    :mod:`stripe_billing_router`.  This function exists to preserve the previous
    public API and now always returns ``True``.
    """

    return True


def get_balance(bot_id: str, *, test_mode: bool = False) -> float:
    """Return available balance for ``bot_id``.

    ``test_mode`` is accepted for backwards compatibility but ignored.
    """

    return _router_get_balance(bot_id)


def charge(
    amount: float,
    bot_id: str,
    *,
    description: str = "",
    test_mode: bool = False,
) -> str:
    """Charge ``amount`` against ``bot_id`` and return the Stripe status.

    ``test_mode`` is accepted for backwards compatibility but ignored.
    """

    resp: dict[str, Any] = init_charge(bot_id, amount, description)
    return resp.get("status", "unknown")


__all__ = ["is_enabled", "get_balance", "charge"]
