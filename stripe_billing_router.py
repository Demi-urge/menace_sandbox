"""Stripe billing router for mapping bots to Stripe products and customers.

The router contains the Stripe API keys as module level constants and exposes
helpers that resolve a routing table to determine which Stripe product, price
and customer should be used for a given bot.  A fatal exception is raised if
the keys are missing or no routing rule matches the supplied bot.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from vault_secret_provider import VaultSecretProvider

try:  # optional dependency
    import stripe  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    stripe = None  # type: ignore
    logging.getLogger(__name__).warning("stripe library unavailable: %s", exc)

logger = logging.getLogger(__name__)


def _load_key(name: str, prefix: str) -> str:
    """Fetch a Stripe key from env or the secret vault and validate it."""

    provider = VaultSecretProvider()
    key = os.getenv(name.upper()) or provider.get(name)
    if not key:
        logger.error("Stripe API keys must be configured and non-empty")
        raise RuntimeError("Stripe API keys must be configured and non-empty")
    if not key.startswith(prefix):
        logger.error("Invalid Stripe API key format for %s", name)
        raise RuntimeError("Invalid Stripe API key format")
    if key.startswith(f"{prefix}test"):
        logger.error("Test mode Stripe API keys are not permitted for %s", name)
        raise RuntimeError("Test mode Stripe API keys are not permitted")
    return key


STRIPE_SECRET_KEY = _load_key("stripe_secret_key", "sk_")
STRIPE_PUBLIC_KEY = _load_key("stripe_public_key", "pk_")

# Base routing rules organised by region -> domain -> bot -> category
# Each leaf mapping contains identifiers used for billing.  The default region
# provides backwards compatible behaviour for callers that do not specify a
# region override.
ROUTING_MAP: dict[
    str, dict[str, dict[str, dict[str, dict[str, str]]]]
] = {
    "default": {
        "finance": {
            "finance_router_bot": {
                "monetization": {
                    "product_id": "prod_finance_router",
                    "price_id": "price_finance_standard",
                    "customer_id": "cus_finance_default",
                }
            }
        }
    }
}

# Legacy alias to maintain compatibility with existing imports and startup
# checks.  ``BILLING_RULES`` references the default region map so updates are
# reflected automatically.
BILLING_RULES = ROUTING_MAP.setdefault("default", {})

# Overrides: (region, domain, name, category, key, value) -> route updates
OVERRIDES: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}


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


def register_route(
    domain: str,
    name: str,
    category: str,
    route: Mapping[str, str],
    *,
    region: str = "default",
) -> None:
    """Register or update a base routing rule.

    The rule is stored within :data:`ROUTING_MAP` under the specified region.
    """

    ROUTING_MAP.setdefault(region, {}).setdefault(domain, {}).setdefault(name, {})[
        category
    ] = dict(route)


def register_override(
    domain: str,
    name: str,
    category: str,
    *,
    key: str,
    value: str,
    route: Mapping[str, str],
    region: str = "default",
) -> None:
    """Register a routing override for a specific bot and qualifier."""
    OVERRIDES[(region, domain, name, category, key, value)] = dict(route)


def _client(api_key: str):
    if not api_key:
        logger.error("Attempted to initialise Stripe client without API key")
        raise RuntimeError("Stripe API key must be configured and non-empty")
    if api_key.startswith("sk_test"):
        logger.error("Test mode Stripe API keys are not permitted")
        raise RuntimeError("Test mode Stripe API keys are not permitted")
    if stripe is None:
        logger.error("Stripe library unavailable")
        raise RuntimeError("stripe library unavailable")
    stripe.api_key = api_key
    return stripe


def _parse_bot_id(bot_id: str) -> tuple[str, str, str]:
    try:
        domain, name, category = bot_id.split(":", 2)
    except ValueError as exc:  # pragma: no cover - input validation
        logger.error("Invalid bot_id '%s'", bot_id)
        raise ValueError("bot_id must be in 'domain:name:category' format") from exc
    return domain, name, category


def _resolve_route(
    bot_id: str, overrides: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    domain, name, category = _parse_bot_id(bot_id)
    region = "default"
    if overrides and "region" in overrides:
        region = overrides["region"]

    # Collect supported domains from all regions and overrides
    supported_domains: set[str] = set()
    for reg_map in ROUTING_MAP.values():
        supported_domains.update(reg_map.keys())
    supported_domains.update(d for _, d, *_ in OVERRIDES.keys())
    if domain not in supported_domains:
        logger.error("Unsupported billing domain '%s'", domain)
        raise RuntimeError(f"Unsupported billing domain '{domain}'")

    region_map = ROUTING_MAP.get(region) or ROUTING_MAP.get("default", {})
    route = region_map.get(domain, {}).get(name, {}).get(category)
    if overrides:
        for key, value in overrides.items():
            if key == "region":
                continue
            update = OVERRIDES.get((region, domain, name, category, key, value))
            if update is None:
                update = OVERRIDES.get(("default", domain, name, category, key, value))
            if update:
                route = {**route, **update} if route else dict(update)
    if not route:
        logger.error("No billing route configured for bot '%s'", bot_id)
        raise RuntimeError(f"No billing route configured for bot '{bot_id}'")
    route.setdefault("secret_key", STRIPE_SECRET_KEY)
    route.setdefault("public_key", STRIPE_PUBLIC_KEY)
    for strategy in _STRATEGIES:
        route = strategy.apply(bot_id, dict(route))
    secret = route.get("secret_key", "")
    public = route.get("public_key", "")
    if not secret or not public:
        logger.error("Resolved route missing Stripe keys for bot '%s'", bot_id)
        raise RuntimeError("Stripe keys are not configured for the resolved route")
    if secret.startswith("sk_test") or public.startswith("pk_test"):
        logger.error("Test mode Stripe API keys are not permitted for bot '%s'", bot_id)
        raise RuntimeError("Test mode Stripe API keys are not permitted")
    if not secret.startswith("sk_") or not public.startswith("pk_"):
        logger.error("Invalid Stripe API key format for bot '%s'", bot_id)
        raise RuntimeError("Invalid Stripe API key format")
    return route


def charge(
    bot_id: str,
    amount: float,
    description: str | None = None,
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Charge a customer for the given bot."""
    route = _resolve_route(bot_id, overrides)
    client = _client(route["secret_key"])
    params = {
        "amount": int(amount * 100),
        "currency": "usd",
        "description": description or route.get("product_id", ""),
    }
    if customer := route.get("customer_id"):
        params["customer"] = customer
    return client.Charge.create(**params)


# Backward compatibility
initiate_charge = charge
init_charge = charge


def get_balance(
    bot_id: str, *, overrides: Optional[Mapping[str, str]] = None
) -> float:
    """Return available balance for the given bot."""
    route = _resolve_route(bot_id, overrides)
    client = _client(route["secret_key"])
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
    return client.Customer.create(**customer_info)


__all__ = [
    "initiate_charge",
    "init_charge",
    "charge",
    "get_balance",
    "create_customer",
    "register_route",
    "register_override",
    "register_strategy",
    "RouteStrategy",
    "ROUTING_MAP",
    "BILLING_RULES",
]
