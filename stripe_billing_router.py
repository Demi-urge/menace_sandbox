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
from datetime import datetime
from typing import Any, Mapping, Optional

import yaml
from dynamic_path_router import resolve_path

from billing import billing_logger
from billing.billing_ledger import record_payment
from billing.billing_log_db import log_billing_event
from discrepancy_db import DiscrepancyDB
from vault_secret_provider import VaultSecretProvider
import alert_dispatcher
import rollback_manager

try:  # optional dependency
    import stripe  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    stripe = None  # type: ignore
    logging.getLogger(__name__).warning("stripe library unavailable: %s", exc)

logger = logging.getLogger(__name__)


def log_critical_discrepancy(message: str, bot_id: str) -> None:
    """Log a critical discrepancy, alert, and rollback.

    The discrepancy *message* is stored in :class:`DiscrepancyDB` with the
    associated *bot_id*. An alert is dispatched and the latest sandbox changes
    from that bot are rolled back.
    """

    try:
        DiscrepancyDB().log(message, {"bot_id": bot_id})
    except Exception:
        logger.exception("failed to log discrepancy for bot '%s'", bot_id)
    try:  # pragma: no cover - external side effects
        alert_dispatcher.dispatch_alert(
            "critical_discrepancy", severity=5, message=message
        )
    except Exception:
        logger.exception("alert dispatch failed for bot '%s'", bot_id)
    try:  # pragma: no cover - rollback side effects
        rollback_manager.RollbackManager().rollback("latest", requesting_bot=bot_id)
    except Exception:
        logger.exception("rollback failed for bot '%s'", bot_id)


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


def _load_master_account_id() -> str:
    """Return the configured Stripe master account identifier.

    The ID is resolved from, in order of precedence:

    1. ``STRIPE_MASTER_ACCOUNT_ID`` environment variable.
    2. Secret vault entry ``stripe_master_account_id``.
    3. ``master_account_id`` field in the Stripe routing configuration file.
    """

    provider = VaultSecretProvider()
    acc = os.getenv("STRIPE_MASTER_ACCOUNT_ID") or provider.get(
        "stripe_master_account_id"
    )
    if not acc:
        cfg_path = os.getenv("STRIPE_ROUTING_CONFIG") or resolve_path(
            "config/stripe_billing_router.yaml"
        ).as_posix()
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
                acc = data.get("master_account_id")
        except FileNotFoundError:
            acc = None
    if not acc:
        logger.error("Stripe master account ID must be configured and non-empty")
        raise RuntimeError("Stripe master account ID must be configured and non-empty")
    return str(acc)


def _load_allowed_keys() -> set[str]:
    """Return the set of allowed Stripe secret keys.

    Keys are resolved from, in order of precedence:

    1. ``STRIPE_ALLOWED_SECRET_KEYS`` environment variable (comma separated).
    2. Secret vault entry ``stripe_allowed_secret_keys``.
    3. ``allowed_secret_keys`` list in the Stripe routing configuration file.
    """

    provider = VaultSecretProvider()
    raw = os.getenv("STRIPE_ALLOWED_SECRET_KEYS") or provider.get(
        "stripe_allowed_secret_keys"
    )
    if not raw:
        cfg_path = os.getenv("STRIPE_ROUTING_CONFIG") or resolve_path(
            "config/stripe_billing_router.yaml"
        ).as_posix()
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
                keys = data.get("allowed_secret_keys")
                if isinstance(keys, list):
                    raw = ",".join(keys)
        except FileNotFoundError:
            raw = None
    if raw:
        return {k.strip() for k in str(raw).split(",") if k.strip()}
    return {STRIPE_SECRET_KEY}


MASTER_ACCOUNT_ID = _load_master_account_id()
ALLOWED_SECRET_KEYS = _load_allowed_keys()

_CONFIG_ENV = "STRIPE_ROUTING_CONFIG"
_DEFAULT_CONFIG = resolve_path("config/stripe_billing_router.yaml").as_posix()


def _load_routing_table(path: str) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Return routing rules loaded from ``path``.

    The configuration is expected to be a nested mapping in the form

    ``domain -> region -> business_category -> bot_name -> route``.

    The file may be JSON or YAML based on its extension. Missing files result in
    an empty routing table with a warning.  Each route must include the
    identifiers ``product_id``, ``price_id`` and ``customer_id`` with non-empty
    string values.  A :class:`RuntimeError` is raised if any route is
    misconfigured.
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
    required = {"product_id", "price_id", "customer_id"}
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
                    missing = required.difference(route.keys())
                    if missing:
                        logger.error(
                            "Billing route %s/%s/%s/%s missing required keys: %s",
                            domain,
                            region,
                            business_category,
                            bot_name,
                            ", ".join(sorted(missing)),
                        )
                        raise RuntimeError(
                            "Billing route missing required keys: "
                            + ", ".join(sorted(missing))
                        )
                    for key, value in route.items():
                        if not isinstance(value, str) or not value.strip():
                            logger.error(
                                "Billing route %s/%s/%s/%s has empty value for %s",
                                domain,
                                region,
                                business_category,
                                bot_name,
                                key,
                            )
                            raise RuntimeError(
                                f"Billing route requires non-empty string for {key}"
                            )
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


def _get_account_id(api_key: str) -> str | None:
    """Return the Stripe account ID associated with ``api_key``."""

    try:
        client = _client(api_key)
    except Exception:
        return None
    try:
        if client:
            acct = client.Account.retrieve()
        else:
            acct = stripe.Account.retrieve(api_key=api_key)
        if isinstance(acct, Mapping):
            return str(acct.get("id"))
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.exception("Stripe account retrieval failed: %s", exc)
    return None


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


def _validate_account(api_key: str, expected_account_id: str) -> bool:
    """Return ``True`` if ``api_key`` belongs to ``expected_account_id``."""

    account_id = _get_account_id(api_key)
    return account_id == expected_account_id


def _critical_discrepancy(
    bot_id: str,
    route: Mapping[str, str],
    message: str,
    *,
    destination: str | None = None,
) -> None:
    """Handle critical billing discrepancies with alerting and rollback."""

    timestamp_ms = int(time.time() * 1000)
    billing_logger.log_event(
        id=None,
        action_type="discrepancy",
        amount=None,
        currency=route.get("currency"),
        timestamp_ms=timestamp_ms,
        user_email=route.get("user_email"),
        bot_id=bot_id,
        destination_account=destination or route.get("secret_key"),
        raw_event_json=None,
        error=True,
    )
    try:  # pragma: no cover - alerting/rollback side effects
        alert_dispatcher.dispatch_alert(
            "critical_discrepancy", severity=5, message=message
        )
    except Exception:
        logger.exception("alert dispatch failed for bot '%s'", bot_id)
    try:
        rollback_manager.RollbackManager().auto_rollback(bot_id, [bot_id])
    except Exception:
        logger.exception("rollback failed for bot '%s'", bot_id)
    raise RuntimeError("critical_discrepancy")


def _verify_secret_key(bot_id: str, route: Mapping[str, str]) -> None:
    key = route.get("secret_key")
    if not key or key not in ALLOWED_SECRET_KEYS:
        _critical_discrepancy(
            bot_id,
            route,
            f"Secret key '{key}' not in allowed list",
            destination=key,
        )
    if not _validate_account(key, MASTER_ACCOUNT_ID):
        acct = _get_account_id(key)
        _critical_discrepancy(
            bot_id,
            route,
            f"Account '{acct}' does not match master account",
            destination=acct,
        )


def _verify_master_account(
    bot_id: str,
    route: Mapping[str, str],
    event: Mapping[str, Any] | None,
    api_key: str,
) -> None:
    destination = None
    if isinstance(event, Mapping):
        destination = (
            event.get("on_behalf_of")
            or event.get("account")
            or (event.get("transfer_data") or {}).get("destination")
        )
    if destination is None:
        destination = _get_account_id(api_key)
    if destination != MASTER_ACCOUNT_ID:
        _critical_discrepancy(
            bot_id,
            route,
            f"Destination account '{destination}' does not match master account",
            destination=destination,
        )


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
    route.setdefault("currency", "usd")
    for strategy in _STRATEGIES:
        route = strategy.apply(bot_id, dict(route))
    route.setdefault("currency", "usd")
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
    _verify_secret_key(bot_id, route)
    api_key = route["secret_key"]
    client = _client(api_key)

    price = price_id or route.get("price_id")
    customer = route.get("customer_id")
    description = description or route.get("product_id", "")
    currency = route.get("currency", "usd")
    user_email = route.get("user_email")

    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    amt: float | None = None

    try:
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

        if price:
            if not customer:
                logger.error(
                    "customer_id required for price based billing for bot '%s'", bot_id
                )
                raise RuntimeError("customer_id required for price based billing")
            idempotency_key = f"{bot_id}-{amt if amt is not None else price}-{timestamp_ms}"
            item_params = {
                "customer": customer,
                "price": price,
                "idempotency_key": idempotency_key,
            }
            if client:
                client.InvoiceItem.create(**item_params)
                invoice = client.Invoice.create(
                    customer=customer,
                    description=description,
                    idempotency_key=idempotency_key,
                )
                event = client.Invoice.pay(
                    invoice["id"], idempotency_key=idempotency_key
                )
            else:
                stripe.InvoiceItem.create(api_key=api_key, **item_params)
                invoice = stripe.Invoice.create(
                    api_key=api_key,
                    customer=customer,
                    description=description,
                    idempotency_key=idempotency_key,
                )
                event = stripe.Invoice.pay(
                    invoice["id"], api_key=api_key, idempotency_key=idempotency_key
                )
            _verify_master_account(bot_id, route, event, api_key)
            return event

        if amt is None:
            logger.error(
                "amount must be provided when price_id is not supplied for bot '%s'", bot_id
            )
            raise ValueError("amount required when price_id is not provided")

        idempotency_key = f"{bot_id}-{amt}-{timestamp_ms}"
        params = {
            "amount": int(amt * 100),
            "currency": currency,
            "description": description,
            "idempotency_key": idempotency_key,
        }
        if customer:
            params["customer"] = customer
        if client:
            event = client.PaymentIntent.create(**params)
        else:
            event = stripe.PaymentIntent.create(api_key=api_key, **params)
        _verify_master_account(bot_id, route, event, api_key)
        return event
    finally:
        destination = None
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
            )
        if destination is None:
            destination = route.get("secret_key")

        logged_amount = amt
        if logged_amount is None and isinstance(event, Mapping):
            possible = event.get("amount") or event.get("amount_paid")
            if possible is not None:
                try:
                    logged_amount = float(possible) / 100.0
                except (TypeError, ValueError):
                    logged_amount = None

        raw_json = None
        if isinstance(event, Mapping):
            try:
                raw_json = json.dumps(event)
            except Exception:  # pragma: no cover - serialization issues
                raw_json = None

        email = user_email
        if event is not None and customer:
            try:
                cust = (
                    client.Customer.retrieve(customer)
                    if client
                    else stripe.Customer.retrieve(customer, api_key=api_key)
                )
                possible_email = None
                if isinstance(cust, Mapping):
                    possible_email = cust.get("email")
                if possible_email:
                    email = possible_email
            except Exception:  # pragma: no cover - best effort
                pass

        billing_logger.log_event(
            id=event.get("id") if isinstance(event, Mapping) else None,
            action_type="charge",
            amount=logged_amount,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            raw_event_json=raw_json,
            error=False,
        )
        record_payment(
            "charge",
            logged_amount,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
        )
        if event is not None:
            log_billing_event(
                "charge",
                bot_id=bot_id,
                amount=logged_amount,
                currency=currency,
                user_email=email,
                destination_account=destination,
                stripe_key=api_key,
                ts=datetime.utcnow().isoformat(),
            )


# Backward compatibility
initiate_charge = charge
init_charge = charge


def get_balance(
    bot_id: str, *, overrides: Optional[Mapping[str, str]] = None
) -> float:
    """Return available balance for the given bot."""
    route = _resolve_route(bot_id, overrides)
    _verify_secret_key(bot_id, route)
    api_key = route["secret_key"]
    client = _client(api_key)
    try:
        if client:
            bal = client.Balance.retrieve()
        else:
            bal = stripe.Balance.retrieve(api_key=api_key)
        _verify_master_account(bot_id, route, bal, api_key)
        amount = bal.get("available", [{"amount": 0}])[0]["amount"] / 100.0
        return float(amount)
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.exception("Stripe balance retrieval failed: %s", exc)
        raise RuntimeError("Stripe balance retrieval failed") from exc


def create_customer(
    bot_id: str,
    customer_info: Mapping[str, Any],
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Create a new Stripe customer for the given bot."""
    route = _resolve_route(bot_id, overrides)
    _verify_secret_key(bot_id, route)
    api_key = route["secret_key"]
    client = _client(api_key)
    if client:
        event = client.Customer.create(**customer_info)
    else:
        event = stripe.Customer.create(api_key=api_key, **customer_info)
    _verify_master_account(bot_id, route, event, api_key)
    return event


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
    _verify_secret_key(bot_id, route)
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
    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    try:
        if client:
            event = client.Subscription.create(**sub_params)
        else:
            event = stripe.Subscription.create(api_key=api_key, **sub_params)
        _verify_master_account(bot_id, route, event, api_key)
        return event
    finally:
        currency = route.get("currency")
        email = route.get("user_email")
        destination = None
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
            )
        if destination is None:
            destination = route.get("secret_key")
        raw_json = None
        if isinstance(event, Mapping):
            try:
                raw_json = json.dumps(event)
            except Exception:  # pragma: no cover - serialization issues
                raw_json = None

        if event is not None and customer:
            try:
                cust = (
                    client.Customer.retrieve(customer)
                    if client
                    else stripe.Customer.retrieve(customer, api_key=api_key)
                )
                possible_email = None
                if isinstance(cust, Mapping):
                    possible_email = cust.get("email")
                if possible_email:
                    email = possible_email
            except Exception:  # pragma: no cover - best effort
                pass

        billing_logger.log_event(
            id=event.get("id") if isinstance(event, Mapping) else None,
            action_type="subscription",
            amount=None,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            raw_event_json=raw_json,
            error=False,
        )
        record_payment(
            "subscription",
            None,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
        )
        if event is not None:
            sub_amount = None
            if price:
                try:
                    price_obj = (
                        client.Price.retrieve(price)
                        if client
                        else stripe.Price.retrieve(price, api_key=api_key)
                    )
                    unit = None
                    if isinstance(price_obj, Mapping):
                        unit = price_obj.get("unit_amount")
                    if unit is not None:
                        sub_amount = float(unit) / 100.0
                except Exception:  # pragma: no cover - best effort
                    pass
            log_billing_event(
                "subscription",
                bot_id=bot_id,
                amount=sub_amount,
                currency=currency,
                user_email=email,
                destination_account=destination,
                stripe_key=api_key,
                ts=datetime.utcnow().isoformat(),
            )


def refund(
    bot_id: str,
    charge_id: str,
    *,
    amount: float | None = None,
    user_email: str | None = None,
    overrides: Optional[Mapping[str, str]] = None,
    **params: Any,
) -> dict[str, Any]:
    """Refund a payment for the given bot using ``charge_id``."""

    route = _resolve_route(bot_id, overrides)
    _verify_secret_key(bot_id, route)
    api_key = route["secret_key"]
    client = _client(api_key)
    refund_params: dict[str, Any] = {"charge": charge_id, **params}
    if amount is not None:
        try:
            refund_params["amount"] = int(float(amount) * 100)
        except (TypeError, ValueError):
            logger.error(
                "amount must be a positive, non-zero float for bot '%s'", bot_id
            )
            raise ValueError("amount must be a positive, non-zero float")
        if refund_params["amount"] <= 0:
            logger.error(
                "amount must be a positive, non-zero float for bot '%s'", bot_id
            )
            raise ValueError("amount must be a positive, non-zero float")
    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    try:
        if client:
            event = client.Refund.create(**refund_params)
        else:
            event = stripe.Refund.create(api_key=api_key, **refund_params)
        _verify_master_account(bot_id, route, event, api_key)
        return event
    finally:
        currency = route.get("currency")
        email = user_email or route.get("user_email")
        destination = None
        logged_amount: float | None = None
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
            )
            possible = event.get("amount")
            if possible is not None:
                try:
                    logged_amount = float(possible) / 100.0
                except (TypeError, ValueError):
                    logged_amount = None
        if destination is None:
            destination = route.get("secret_key")
        raw_json = None
        if isinstance(event, Mapping):
            try:
                raw_json = json.dumps(event)
            except Exception:  # pragma: no cover - serialization issues
                raw_json = None
        billing_logger.log_event(
            id=event.get("id") if isinstance(event, Mapping) else None,
            action_type="refund",
            amount=logged_amount,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            raw_event_json=raw_json,
            error=False,
        )
        record_payment(
            "refund",
            logged_amount,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
        )


def create_checkout_session(
    bot_id: str,
    line_items: list[Mapping[str, Any]],
    *,
    amount: float | None = None,
    user_email: str | None = None,
    overrides: Optional[Mapping[str, str]] = None,
    **params: Any,
) -> dict[str, Any]:
    """Create a Stripe Checkout session for the given bot."""

    route = _resolve_route(bot_id, overrides)
    _verify_secret_key(bot_id, route)
    api_key = route["secret_key"]
    client = _client(api_key)
    session_params = {"line_items": list(line_items), **params}
    if "customer" not in session_params and route.get("customer_id"):
        session_params["customer"] = route["customer_id"]
    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    try:
        if client:
            event = client.checkout.Session.create(**session_params)
        else:
            event = stripe.checkout.Session.create(api_key=api_key, **session_params)
        _verify_master_account(bot_id, route, event, api_key)
        return event
    finally:
        currency = route.get("currency")
        email = user_email or route.get("user_email")
        destination = None
        logged_amount: float | None = amount
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
            )
            if logged_amount is None:
                possible = (
                    event.get("amount_total")
                    or event.get("amount")
                    or event.get("amount_subtotal")
                )
                if possible is not None:
                    try:
                        logged_amount = float(possible) / 100.0
                    except (TypeError, ValueError):
                        logged_amount = None
        if destination is None:
            destination = route.get("secret_key")
        raw_json = None
        if isinstance(event, Mapping):
            try:
                raw_json = json.dumps(event)
            except Exception:  # pragma: no cover - serialization issues
                raw_json = None
        billing_logger.log_event(
            id=event.get("id") if isinstance(event, Mapping) else None,
            action_type="checkout_session",
            amount=logged_amount,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            raw_event_json=raw_json,
            error=False,
        )
        record_payment(
            "checkout_session",
            logged_amount,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
        )


__all__ = [
    "initiate_charge",
    "init_charge",
    "charge",
    "get_balance",
    "create_customer",
    "create_subscription",
    "refund",
    "create_checkout_session",
    "register_route",
    "register_override",
    "register_strategy",
    "RouteStrategy",
    "ROUTING_TABLE",
    "BILLING_RULES",
]
