"""Stripe billing router for mapping bots to Stripe products and customers.

The router contains the Stripe API keys as module level constants and exposes
helpers that resolve a routing table to determine which Stripe product, price
and customer should be used for a given bot.  Routing rules are stored in a
dictionary keyed by ``(domain, region, business_category, bot_name)`` tuples
which makes it easy to register new combinations or override existing ones
dynamically.  A fatal exception is raised if the keys are missing or no routing
rule matches the supplied bot.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping, Optional

import yaml
from dynamic_path_router import resolve_path

from vault_secret_provider import VaultSecretProvider

try:  # optional dependency
    import stripe  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    stripe = None  # type: ignore
    logging.getLogger(__name__).warning("stripe library unavailable: %s", exc)

logger = logging.getLogger(__name__)


def _validate_no_api_keys(mapping: Mapping[str, str]) -> None:
    """Ensure a route mapping does not attempt to override Stripe keys."""

    forbidden = {"secret_key", "public_key"}
    present = forbidden.intersection(mapping.keys())
    if present:
        logger.error("Stripe API keys cannot be supplied in routes: %s", present)
        raise ValueError("Stripe API keys cannot be overridden")


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

_CONFIG_ENV = "STRIPE_ROUTING_CONFIG"
_DEFAULT_CONFIG = resolve_path("config/stripe_billing_router.yaml").as_posix()


def _load_routing_table(path: str) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Return routing rules loaded from ``path``.

    The configuration is expected to be a nested mapping in the form

    ``domain -> region -> business_category -> bot_name -> route``.

    The file may be JSON or YAML based on its extension. Missing files result in
    an empty routing table with a warning.
    """

    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith(".json"):
                data = json.load(fh)
            else:
                data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Stripe routing config missing at %%s", path)
        return {}

    table: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for domain, regions in (data or {}).items():
        if not isinstance(regions, Mapping):
            continue
        for region, categories in regions.items():
            if not isinstance(categories, Mapping):
                continue
            for business_category, bots in categories.items():
                if not isinstance(bots, Mapping):
                    continue
                for bot_name, route in bots.items():
                    if not isinstance(route, Mapping):
                        continue
                    _validate_no_api_keys(route)
                    table[(str(domain), str(region), str(business_category), str(bot_name))] = {
                        str(k): str(v) for k, v in route.items()
                    }
    return table


_ROUTING_CONFIG_PATH = os.getenv(_CONFIG_ENV, _DEFAULT_CONFIG)

# Base routing rules keyed by (domain, region, business_category, bot_name).
# Each value contains identifiers used for billing.  The default region provides
# backwards compatible behaviour for callers that do not specify a region
# override.
ROUTING_TABLE: dict[tuple[str, str, str, str], dict[str, str]] = _load_routing_table(
    _ROUTING_CONFIG_PATH
)

# Legacy alias maintained for backwards compatibility with existing imports.
BILLING_RULES = ROUTING_TABLE

# Overrides: (domain, region, business_category, bot_name, key, value) -> route
# updates
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
    business_category: str,
    bot_name: str,
    route: Mapping[str, str],
    *,
    domain: str = "stripe",
    region: str = "default",
) -> None:
    """Register or update a base routing rule.

    ``domain`` is optional and defaults to ``"stripe"`` for backwards
    compatibility.  The rule is stored within :data:`ROUTING_TABLE` under the
    specified domain and region.
    """

    _validate_no_api_keys(route)
    ROUTING_TABLE[(domain, region, business_category, bot_name)] = dict(route)


def register_override(rule: Mapping[str, Any]) -> None:
    """Register a routing override for a specific bot and qualifier.

    The ``rule`` mapping requires the keys ``business_category``, ``bot_name``,
    ``key``, ``value`` and ``route``. Optional ``domain`` defaults to
    ``"stripe"`` and ``region`` defaults to ``"default"``.
    """

    domain = rule.get("domain", "stripe")
    region = rule.get("region", "default")
    business_category = rule["business_category"]
    bot_name = rule["bot_name"]
    key = rule["key"]
    value = rule["value"]
    route = rule["route"]
    _validate_no_api_keys(route)
    OVERRIDES[(domain, region, business_category, bot_name, key, value)] = dict(route)


def _client(api_key: str):
    """Return a per-request Stripe client without mutating global state."""

    if not api_key:
        logger.error("Attempted to initialise Stripe client without API key")
        raise RuntimeError("Stripe API key must be configured and non-empty")
    if api_key.startswith("sk_test"):
        logger.error("Test mode Stripe API keys are not permitted")
        raise RuntimeError("Test mode Stripe API keys are not permitted")
    if stripe is None:
        logger.error("Stripe library unavailable")
        raise RuntimeError("stripe library unavailable")
    if hasattr(stripe, "StripeClient"):
        return stripe.StripeClient(api_key)
    return None


def _parse_bot_id(bot_id: str) -> tuple[str, str, str]:
    """Return ``(domain, business_category, bot_name)`` from ``bot_id``.

    ``bot_id`` can be provided either as ``"domain:category:bot"`` or the
    legacy ``"category:bot"`` format.  In the latter case the domain defaults to
    ``"stripe"`` for backwards compatibility.
    """

    parts = bot_id.split(":")
    if len(parts) == 2:
        business_category, bot_name = parts
        domain = "stripe"
    elif len(parts) == 3:
        domain, business_category, bot_name = parts
    else:
        logger.error("Invalid bot_id '%s'", bot_id)
        raise ValueError(
            "bot_id must be in 'domain:business_category:bot_name' format"
        )
    return domain, business_category, bot_name


def _resolve_route(
    bot_id: str, overrides: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    domain, business_category, bot_name = _parse_bot_id(bot_id)
    region = "default"
    if overrides and "region" in overrides:
        region = overrides["region"]

    # Collect supported domains from all routes and overrides
    supported_domains: set[str] = {d for d, _, _, _ in ROUTING_TABLE.keys()}
    supported_domains.update(d for d, _, _, _, _, _ in OVERRIDES.keys())
    if not domain or domain not in supported_domains:
        logger.error("Unsupported billing domain '%s'", domain)
        raise RuntimeError(f"Unsupported billing domain '{domain}'")

    # Collect supported business categories from all routes and overrides
    supported_categories: set[str] = {bc for _, _, bc, _ in ROUTING_TABLE.keys()}
    supported_categories.update(bc for _, _, bc, _, _, _ in OVERRIDES.keys())
    if business_category not in supported_categories:
        logger.error("Unsupported billing category '%s'", business_category)
        raise RuntimeError(
            f"Unsupported billing category '{business_category}'"
        )

    route = ROUTING_TABLE.get((domain, region, business_category, bot_name))
    if route:
        route = dict(route)
        _validate_no_api_keys(route)
    if overrides:
        for key, value in overrides.items():
            if key == "region":
                continue
            if key in {"secret_key", "public_key"}:
                logger.error("Stripe API keys cannot be overridden for bot '%s'", bot_id)
                raise RuntimeError("Stripe API keys cannot be overridden")
            update = OVERRIDES.get(
                (domain, region, business_category, bot_name, key, value)
            )
            if update is None:
                update = OVERRIDES.get(
                    (domain, "default", business_category, bot_name, key, value)
                )
            if update:
                _validate_no_api_keys(update)
                route = {**route, **update} if route else dict(update)
    if not route:
        logger.error("No billing route configured for bot '%s'", bot_id)
        raise RuntimeError(f"No billing route configured for bot '{bot_id}'")
    _validate_no_api_keys(route)
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
    amount: float | None = None,
    description: str | None = None,
    *,
    price_id: str | None = None,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Charge a customer for the given bot.

    ``price_id`` can be supplied to bill using a preconfigured Stripe Price.  In
    that case an Invoice Item is created for the price and immediately charged
    via the Invoice API.  When ``price_id`` is omitted, a one‑off charge is
    performed using the PaymentIntent API and ``amount`` must be provided.
    """

    route = _resolve_route(bot_id, overrides)
    api_key = route["secret_key"]
    client = _client(api_key)

    price = price_id or route.get("price_id")
    customer = route.get("customer_id")
    description = description or route.get("product_id", "")

    amt: float | None = None
    if amount is not None:
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            logger.error(
                "amount must be a positive, non-zero float for bot '%s'", bot_id
            )
            raise ValueError("amount must be a positive, non-zero float")
        if amt <= 0:
            logger.error(
                "amount must be a positive, non-zero float for bot '%s'", bot_id
            )
            raise ValueError("amount must be a positive, non-zero float")

    timestamp_ms = int(time.time() * 1000)

    if price:
        if not customer:
            logger.error(
                "customer_id required for price based billing for bot '%s'", bot_id
            )
            raise RuntimeError("customer_id required for price based billing")
        idempotency_key = f"{bot_id}-{amt if amt is not None else price}-{timestamp_ms}"
        item_params = {"customer": customer, "price": price, "idempotency_key": idempotency_key}
        if client:
            client.InvoiceItem.create(**item_params)
            invoice = client.Invoice.create(
                customer=customer, description=description, idempotency_key=idempotency_key
            )
            return client.Invoice.pay(invoice["id"], idempotency_key=idempotency_key)
        stripe.InvoiceItem.create(api_key=api_key, **item_params)
        invoice = stripe.Invoice.create(
            api_key=api_key,
            customer=customer,
            description=description,
            idempotency_key=idempotency_key,
        )
        return stripe.Invoice.pay(
            invoice["id"], api_key=api_key, idempotency_key=idempotency_key
        )

    if amt is None:
        logger.error(
            "amount must be provided when price_id is not supplied for bot '%s'", bot_id
        )
        raise ValueError("amount required when price_id is not provided")

    idempotency_key = f"{bot_id}-{amt}-{timestamp_ms}"
    params = {
        "amount": int(amt * 100),
        "currency": "usd",
        "description": description,
        "idempotency_key": idempotency_key,
    }
    if customer:
        params["customer"] = customer
    if client:
        return client.PaymentIntent.create(**params)
    return stripe.PaymentIntent.create(api_key=api_key, **params)


# Backward compatibility
initiate_charge = charge
init_charge = charge


def get_balance(
    bot_id: str, *, overrides: Optional[Mapping[str, str]] = None
) -> float:
    """Return available balance for the given bot."""
    route = _resolve_route(bot_id, overrides)
    api_key = route["secret_key"]
    client = _client(api_key)
    try:
        if client:
            bal = client.Balance.retrieve()
        else:
            bal = stripe.Balance.retrieve(api_key=api_key)
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
    api_key = route["secret_key"]
    client = _client(api_key)
    if client:
        return client.Customer.create(**customer_info)
    return stripe.Customer.create(api_key=api_key, **customer_info)


def create_subscription(
    bot_id: str,
    *,
    price_id: str | None = None,
    customer_id: str | None = None,
    overrides: Optional[Mapping[str, str]] = None,
    **params: Any,
) -> dict[str, Any]:
    """Create a recurring subscription for the given bot.

    ``price_id`` and ``customer_id`` default to values resolved from the routing
    table.  Additional Stripe ``Subscription.create`` parameters may be supplied
    via ``params``.
    """

    route = _resolve_route(bot_id, overrides)
    api_key = route["secret_key"]
    client = _client(api_key)
    price = price_id or route.get("price_id")
    customer = customer_id or route.get("customer_id")
    if not price or not customer:
        logger.error(
            "price_id and customer_id are required for subscriptions for bot '%s'",
            bot_id,
        )
        raise RuntimeError("price_id and customer_id are required for subscriptions")
    sub_params = {"customer": customer, "items": [{"price": price}], **params}
    if client:
        return client.Subscription.create(**sub_params)
    return stripe.Subscription.create(api_key=api_key, **sub_params)


__all__ = [
    "initiate_charge",
    "init_charge",
    "charge",
    "get_balance",
    "create_customer",
    "create_subscription",
    "register_route",
    "register_override",
    "register_strategy",
    "RouteStrategy",
    "ROUTING_TABLE",
    "BILLING_RULES",
]
