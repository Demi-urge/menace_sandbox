"""Stripe billing router for mapping bots to Stripe products and customers.

This module selects Stripe API keys and product/customer identifiers based on
bot metadata. Keys are obtained from a secure vault provider or fall back to
hard coded production values. A fatal exception is raised if API keys are
missing or no routing rule matches the supplied bot.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from .stripe_handler import is_enabled
from .vault_secret_provider import VaultSecretProvider

try:  # optional dependency
    import stripe  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    stripe = None  # type: ignore
    logging.getLogger(__name__).warning("stripe library unavailable: %s", exc)

logger = logging.getLogger(__name__)

# Retrieve keys from the vault provider with baked‑in fallbacks. These
# placeholders represent production keys and must never be empty.
_vault = VaultSecretProvider()
OFFICIAL_SECRET_KEY = "sk_live_official_placeholder"
OFFICIAL_PUBLIC_KEY = "pk_live_official_placeholder"
STRIPE_SECRET_KEY = _vault.get("stripe_secret_key") or OFFICIAL_SECRET_KEY
STRIPE_PUBLIC_KEY = _vault.get("stripe_public_key") or OFFICIAL_PUBLIC_KEY
if not STRIPE_SECRET_KEY or not STRIPE_PUBLIC_KEY:
    raise RuntimeError("Stripe API keys must be configured and non-empty")

# Base routing rules: (domain, name, category) -> identifiers
BILLING_RULES: dict[tuple[str, str, str], dict[str, str]] = {
    (
        "finance",
        "finance_router_bot",
        "monetization",
    ): {
        "product_id": "prod_finance_router",
        "price_id": "price_finance_standard",
        "customer_id": "cus_finance_default",
    }
}

# Overrides: (domain, name, category, key, value) -> route updates
OVERRIDES: dict[tuple[str, str, str, str, str], dict[str, str]] = {}


class RouteStrategy:
    """Strategy hook allowing dynamic route overrides (e.g., per region)."""

    def apply(self, bot_id: str, route: dict[str, str]) -> dict[str, str]:
        """Return an updated route for ``bot_id``.

        Subclasses can mutate ``route`` or return a new mapping. The default
        implementation returns ``route`` unchanged.
        """

        return route


_STRATEGIES: list[RouteStrategy] = []


def register_strategy(strategy: RouteStrategy) -> None:
    """Register a strategy that can override resolved routes."""

    _STRATEGIES.append(strategy)


def register_override(
    domain: str,
    name: str,
    category: str,
    *,
    key: str,
    value: str,
    route: Mapping[str, str],
) -> None:
    """Register a routing override for a specific bot and qualifier."""
    OVERRIDES[(domain, name, category, key, value)] = dict(route)


def _use_mock() -> bool:
    return os.getenv("MENACE_MODE", "test").lower() != "production"


class _MockCharge:
    @staticmethod
    def create(**kwargs: Any) -> dict[str, Any]:
        logger.info("Mock Stripe charge: %s", kwargs)
        return {"id": "mock_charge", "status": "succeeded"}


class _MockCustomer:
    @staticmethod
    def create(**kwargs: Any) -> dict[str, Any]:
        logger.info("Mock Stripe customer: %s", kwargs)
        return {"id": "mock_customer"}


class _MockStripe:
    api_key: str = ""
    Charge = _MockCharge
    Customer = _MockCustomer


class _MockBalance:
    @staticmethod
    def retrieve() -> dict[str, Any]:
        logger.info("Mock Stripe balance retrieval")
        return {"available": [{"amount": 0}]}


_MockStripe.Balance = _MockBalance


def _client(api_key: str):
    if not is_enabled() or not api_key:
        return None
    if stripe is None or _use_mock():
        return _MockStripe()
    stripe.api_key = api_key
    return stripe


def _parse_bot_id(bot_id: str) -> tuple[str, str, str]:
    try:
        domain, name, category = bot_id.split(":", 2)
    except ValueError as exc:  # pragma: no cover - input validation
        raise ValueError("bot_id must be in 'domain:name:category' format") from exc
    return domain, name, category


def _resolve_route(
    bot_id: str, overrides: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    domain, name, category = _parse_bot_id(bot_id)
    route = BILLING_RULES.get((domain, name, category))
    if overrides:
        for key, value in overrides.items():
            update = OVERRIDES.get((domain, name, category, key, value))
            if update:
                if route:
                    route = {**route, **update}
                else:
                    route = dict(update)
    if not route:
        raise RuntimeError(f"No billing route for bot '{bot_id}'")
    route.setdefault("secret_key", STRIPE_SECRET_KEY)
    route.setdefault("public_key", STRIPE_PUBLIC_KEY)
    for strategy in _STRATEGIES:
        route = strategy.apply(bot_id, dict(route))
    if not route.get("secret_key") or not route.get("public_key"):
        raise RuntimeError("Stripe keys are not configured for the resolved route")
    return route


def init_charge(
    bot_id: str,
    amount: float,
    description: str | None = None,
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Charge a customer for the given bot."""
    route = _resolve_route(bot_id, overrides)
    client = _client(route["secret_key"])
    if client is None:
        raise RuntimeError("Stripe client unavailable or disabled")
    params = {
        "amount": int(amount * 100),
        "currency": "usd",
        "description": description or route.get("product_id", ""),
    }
    if customer := route.get("customer_id"):
        params["customer"] = customer
    if _use_mock():
        params["source"] = "tok_visa"
    return client.Charge.create(**params)


# Backward compatibility
charge = init_charge


def get_balance(
    bot_id: str, *, overrides: Optional[Mapping[str, str]] = None
) -> float:
    """Return available balance for the given bot."""
    route = _resolve_route(bot_id, overrides)
    client = _client(route["secret_key"])
    if client is None:
        logger.info("Stripe balance retrieval skipped: client unavailable")
        return 0.0
    try:
        bal = client.Balance.retrieve()
        amount = bal.get("available", [{"amount": 0}])[0]["amount"] / 100.0
        return float(amount)
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.exception("Stripe balance retrieval failed: %s", exc)
        return 0.0


def create_customer(
    bot_id: str,
    customer_info: Mapping[str, Any],
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Create a new Stripe customer for the given bot."""
    route = _resolve_route(bot_id, overrides)
    client = _client(route["secret_key"])
    if client is None:
        raise RuntimeError("Stripe client unavailable or disabled")
    return client.Customer.create(**customer_info)


__all__ = [
    "init_charge",
    "charge",
    "get_balance",
    "create_customer",
    "register_override",
    "register_strategy",
    "RouteStrategy",
]
