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
from collections.abc import Iterator, MutableSet
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TextIO
import hashlib

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning("PyYAML unavailable: %s", exc)
from dynamic_path_router import resolve_path

from billing import billing_logger
from billing.billing_ledger import record_payment
from billing.billing_log_db import log_billing_event
from billing.stripe_ledger import StripeLedger
try:  # pragma: no cover - fallback for tests or missing deps
    from discrepancy_db import DiscrepancyDB, DiscrepancyRecord
except Exception:  # pragma: no cover - simplified stubs
    from discrepancy_db import DiscrepancyDB  # type: ignore
    from dataclasses import dataclass, field
    from typing import Dict

    @dataclass
    class DiscrepancyRecord:  # type: ignore[override]
        message: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())
        id: int = 0
from vault_secret_provider import VaultSecretProvider
import alert_dispatcher
import rollback_manager
import sandbox_review
import menace_sanity_layer
from menace_sanity_layer import record_payment_anomaly, record_billing_event

try:  # optional dependency
    import stripe  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    stripe = None  # type: ignore
    logging.getLogger(__name__).warning("stripe library unavailable: %s", exc)

try:  # optional dependency
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - optional dependency
    dotenv_values = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _load_yaml_data(handle: TextIO, *, source: str) -> Any:
    """Load YAML content while gracefully handling missing dependencies."""

    if yaml is not None:  # type: ignore[truthy-function]
        try:
            data = yaml.safe_load(handle)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("Failed to parse YAML %s: %s", source, exc)
            return {}
        return data or {}

    text = handle.read()
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "PyYAML not installed; ignoring YAML configuration at %s", source
        )
        return {}
    logger.info(
        "Parsed %s as JSON because PyYAML is unavailable", source
    )
    return parsed or {}


def _load_serialised_mapping(path: str) -> Mapping[str, Any]:
    """Return mapping data loaded from JSON or YAML configuration files."""

    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.lower().endswith(".json"):
                try:
                    data = json.load(fh)
                except json.JSONDecodeError as exc:
                    logger.error("Invalid JSON in %s: %s", path, exc)
                    return {}
            else:
                data = _load_yaml_data(fh, source=path)
    except FileNotFoundError:
        logger.warning("Stripe routing config missing at %s", path)
        return {}

    if not isinstance(data, Mapping):
        logger.warning("Configuration %s is not a mapping; ignoring", path)
        return {}
    return dict(data)


_STRIPE_LEDGER = StripeLedger()
_SKIP_STRIPE_ENV_VAR = "MENACE_SKIP_STRIPE_ROUTER"


def _skip_stripe_verification() -> bool:
    """Return ``True`` when Stripe router verification should be bypassed."""

    value = os.getenv(_SKIP_STRIPE_ENV_VAR)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


class _AllowedSecretKeySet(MutableSet[str]):
    """Lazy wrapper around the configured Stripe secret keys."""

    def __init__(self) -> None:
        self._cache: set[str] | None = None

    def _materialise(self) -> set[str]:
        if self._cache is None:
            if _skip_stripe_verification():
                self._cache = set()
            else:
                self._cache = _load_allowed_keys()
        return self._cache

    def __contains__(self, value: object) -> bool:
        return value in self._materialise()

    def __iter__(self) -> Iterator[str]:
        return iter(self._materialise())

    def __len__(self) -> int:
        return len(self._materialise())

    def add(self, value: str) -> None:
        self._materialise().add(value)

    def discard(self, value: str) -> None:
        self._materialise().discard(value)

    def clear(self) -> None:
        self._materialise().clear()

    def refresh(self) -> None:
        self._cache = None

    def materialise(self) -> set[str]:
        return self._materialise()


ALLOWED_SECRET_KEYS: _AllowedSecretKeySet = _AllowedSecretKeySet()


def _allowed_secret_keys() -> set[str]:
    """Return the cached set of allowed Stripe secret keys."""

    return ALLOWED_SECRET_KEYS.materialise()

# Billing instruction overrides used by :mod:`menace_sanity_layer`.  The
# instructions are cached within that module so we monitor the config file and
# trigger a refresh when it changes.
_BILLING_INSTRUCTIONS_PATH = Path(resolve_path("config/billing_instructions.yaml"))
_BILLING_INSTRUCTIONS_MTIME = 0.0


def _refresh_instruction_cache() -> None:
    """Reload cached billing instructions when the config file changes."""

    global _BILLING_INSTRUCTIONS_MTIME
    try:
        mtime = _BILLING_INSTRUCTIONS_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    if mtime != _BILLING_INSTRUCTIONS_MTIME:
        menace_sanity_layer.refresh_billing_instructions(_BILLING_INSTRUCTIONS_PATH)
        _BILLING_INSTRUCTIONS_MTIME = mtime


def _hash_api_key(key: str) -> str:
    """Return a non-reversible identifier for a Stripe API key."""

    return hashlib.sha256(key.encode()).hexdigest()


def _log_payment(
    action: str,
    bot_id: str,
    amount: float,
    currency: str,
    email: Optional[str],
    account_id: str,
    ts: int,
    charge_id: str | None = None,
) -> None:
    """Persist a payment event to the Stripe ledger."""

    try:  # pragma: no cover - best effort logging
        _STRIPE_LEDGER.log_event(
            action, bot_id, amount, currency, email, account_id, ts, charge_id
        )
    except Exception:
        logger.exception("failed to log payment action '%s' for bot '%s'", action, bot_id)


def log_critical_discrepancy(bot_id: str, message: str) -> None:
    """Record a critical discrepancy, alert, and rollback."""
    _refresh_instruction_cache()
    record_payment_anomaly(
        "critical_discrepancy",
        {"bot_id": bot_id, "message": message},
        "Route all payment events through central handler and log anomalies.",
        severity=5.0,
    )
    rec = DiscrepancyRecord(message=message, metadata={"bot_id": bot_id})
    try:
        DiscrepancyDB().add(rec)
    except Exception:
        logger.exception("failed to record discrepancy for bot '%s'", bot_id)
    try:  # pragma: no cover - external side effects
        alert_dispatcher.dispatch_alert(
            "critical_discrepancy", 5, message, {"bot": bot_id}
        )
    except Exception:
        logger.exception("alert dispatch failed for bot '%s'", bot_id)
    try:  # pragma: no cover - rollback side effects
        rm = rollback_manager.RollbackManager()
    except Exception:
        logger.exception("rollback manager init failed for bot '%s'", bot_id)
    else:
        try:
            rm.rollback("latest", requesting_bot=bot_id)
        except Exception:
            logger.exception("rollback failed for bot '%s'", bot_id)


def _alert_mismatch(
    bot_id: str,
    account_id: str,
    message: str = "Stripe account mismatch",
    amount: float | None = None,
) -> None:
    """Backward-compatible wrapper for critical discrepancy handling."""
    from evolution_lock_flag import trigger_lock

    _refresh_instruction_cache()
    record_payment_anomaly(
        "stripe_account_mismatch",
        {"bot_id": bot_id, "account_id": account_id, "amount": amount},
        "All mismatch events must go through central routing and be logged.",
        severity=5.0,
    )
    record_billing_event(
        "stripe_account_mismatch",
        {"bot_id": bot_id, "account_id": account_id, "amount": amount},
        (
            "Ensure all Stripe operations route through stripe_billing_router "
            "and validate destination accounts."
        ),
    )
    menace_sanity_layer.record_event(
        "account_mismatch",
        {"bot_id": bot_id, "destination_account": account_id, "amount": amount},
    )
    trigger_lock(f"Stripe account mismatch for {bot_id}", severity=5)
    log_critical_discrepancy(bot_id, message)
    # Pause the bot in the sandbox so further actions require review.
    try:
        sandbox_review.pause_bot(bot_id)
    except Exception:  # pragma: no cover - pause is best effort
        logger.exception("failed to pause bot '%s' for review", bot_id)
    timestamp_ms = int(time.time() * 1000)
    billing_logger.log_event(
        error=True,
        action_type="mismatch",
        bot_id=bot_id,
        destination_account=account_id,
        amount=amount,
        timestamp_ms=timestamp_ms,
    )
    log_billing_event(
        "mismatch",
        bot_id=bot_id,
        amount=amount,
        destination_account=account_id,
    )
    return


def _validate_destination(bot_id: str, destination_account: str) -> None:
    """Ensure ``destination_account`` matches the master Stripe account.

    The Stripe API may return events with a ``destination`` or ``on_behalf_of``
    account.  This helper validates that the account matches
    ``STRIPE_MASTER_ACCOUNT_ID`` and raises an exception if it does not.
    """

    if destination_account and destination_account != STRIPE_MASTER_ACCOUNT_ID:
        logger.error(
            "Stripe destination mismatch for bot '%s': %s",
            bot_id,
            destination_account,
        )
        _alert_mismatch(bot_id, destination_account)
        raise RuntimeError("Stripe account mismatch")


def _ensure_destination_account(
    bot_id: str, destination: str | None, amount: float | None = None
) -> None:
    """Validate that ``destination`` matches ``STRIPE_MASTER_ACCOUNT_ID``.

    If the destination account does not match the registered master account an
    alert is dispatched, the event is marked for rollback, and a ``RuntimeError``
    is raised.  ``amount`` is optional and used only for logging/rollback
    purposes when a partial charge may have occurred.
    """

    if destination and destination != STRIPE_MASTER_ACCOUNT_ID:
        _alert_mismatch(bot_id, destination, amount=amount)
        try:  # pragma: no cover - best effort rollback signalling
            rollback_manager.RollbackManager().log_healing_action(
                bot_id, "stripe_destination_mismatch"
            )
        except Exception:
            logger.exception("failed to flag rollback for bot '%s'", bot_id)
        raise RuntimeError("Stripe account mismatch")


def validate_webhook_account(event: Mapping[str, Any]) -> bool:
    """Validate that a Stripe webhook belongs to the master account.

    The webhook payload may indicate the originating account in a number of
    fields.  This helper extracts the account identifier from common locations
    and ensures it matches :data:`STRIPE_MASTER_ACCOUNT_ID`.  If the extracted
    account differs, a mismatch alert is triggered and ``False`` is returned so
    callers can halt further processing.

    Parameters
    ----------
    event:
        The deserialised webhook event payload.

    Returns
    -------
    bool
        ``True`` when the account matches the registered master account or no
        account information is present.  ``False`` when a mismatch is detected.
    """

    obj = {}
    try:
        obj = (event.get("data") or {}).get("object") or {}
    except Exception:
        obj = {}

    account_id = (
        event.get("account")
        or obj.get("on_behalf_of")
        or (obj.get("transfer_data") or {}).get("destination")
    )

    if not account_id:
        transfer = obj.get("transfer")
        if isinstance(transfer, Mapping):
            account_id = transfer.get("destination") or (
                (transfer.get("metadata") or {}).get("destination_account")
            )
        metadata = obj.get("metadata")
        if not account_id and isinstance(metadata, Mapping):
            account_id = (
                metadata.get("destination_account")
                or metadata.get("account")
                or metadata.get("stripe_account")
            )

    account_id = str(account_id) if account_id else ""
    if account_id and account_id != STRIPE_MASTER_ACCOUNT_ID:
        metadata = obj.get("metadata")
        bot_id = "unknown"
        if isinstance(metadata, Mapping):
            bot_id = str(
                metadata.get("bot_id") or metadata.get("bot") or "unknown"
            )
        _alert_mismatch(bot_id, account_id)
        return False

    return True


def _validate_no_api_keys(mapping: Mapping[str, str]) -> None:
    """Ensure a route mapping does not attempt to override Stripe keys."""

    forbidden = {"secret_key", "public_key"}
    present = forbidden.intersection(mapping.keys())
    if present:
        logger.error("Stripe API keys cannot be supplied in routes: %s", present)
        raise ValueError("Stripe API keys cannot be overridden")


def _load_env_file_for_stripe_keys() -> None:
    """Populate Stripe key environment variables from a ``.env`` file."""

    if dotenv_values is None:
        return

    env_path = os.getenv("MENACE_ENV_FILE") or ".env"
    try:
        values = dotenv_values(env_path)
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("failed to load dotenv file: %s", env_path)
        return

    if not values:
        return

    for key in ("STRIPE_SECRET_KEY", "STRIPE_PUBLIC_KEY"):
        if key in os.environ:
            continue
        value = values.get(key)
        if value:
            os.environ[key] = value


def _load_key(name: str, prefix: str) -> str:
    """Fetch a Stripe key from env or the secret vault and validate it."""

    provider = VaultSecretProvider()
    raw = os.getenv(name.upper()) or provider.get(name)
    placeholder = f"{prefix}sandbox-placeholder"
    if not raw:
        logger.warning(
            "Stripe API key '%s' missing; using sandbox placeholder", name
        )
        os.environ.setdefault(name.upper(), placeholder)
        return placeholder
    key = str(raw).strip()
    if not key.startswith(prefix) or key.startswith(f"{prefix}test"):
        logger.warning(
            "Stripe API key '%s' invalid or test-mode; using sandbox placeholder", name
        )
        os.environ[name.upper()] = placeholder
        return placeholder
    return key


_load_env_file_for_stripe_keys()

STRIPE_SECRET_KEY = _load_key("stripe_secret_key", "sk_")
STRIPE_PUBLIC_KEY = _load_key("stripe_public_key", "pk_")

# The identifier for the platform's primary Stripe account is intentionally
# immutable and must never be sourced from environment variables or secret
# storage. All billing operations are performed on behalf of this account.
# There is deliberately no environment or secret-based fallback so this value
# cannot be overridden at runtime.
STRIPE_MASTER_ACCOUNT_ID = "acct_1H123456789ABCDEF"

# Backwards compatibility aliases
STRIPE_REGISTERED_ACCOUNT_ID = STRIPE_MASTER_ACCOUNT_ID
STRIPE_REGISTERED_ACCOUNT = STRIPE_MASTER_ACCOUNT_ID
STRIPE_MASTER_ACCOUNT = STRIPE_MASTER_ACCOUNT_ID


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
        data = _load_serialised_mapping(cfg_path)
        keys = data.get("allowed_secret_keys")
        if isinstance(keys, list):
            raw = ",".join(keys)
    keys = (
        {k.strip() for k in str(raw).split(",") if k.strip()}
        if raw
        else {STRIPE_SECRET_KEY}
    )
    allowed: set[str] = set()
    for key in keys:
        account_id = _get_account_id(key) or ""
        if account_id == STRIPE_MASTER_ACCOUNT_ID:
            allowed.add(key)
        elif stripe is None or key.endswith("sandbox-placeholder"):
            logger.info(
                "Allowing Stripe key %s without verification in sandbox mode",
                _hash_api_key(key)[:12],
            )
            allowed.add(key)
        else:  # pragma: no cover - defensive logging
            logger.error(
                "Ignoring Stripe key for foreign account %s", account_id or "unknown"
            )
    return allowed


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

    data = _load_serialised_mapping(path)
    if not data:
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
                        if key == "account_id" and not value.startswith("acct_"):
                            logger.error(
                                "Billing route %s/%s/%s/%s has invalid account_id",
                                domain,
                                region,
                                business_category,
                                bot_name,
                            )
                            raise RuntimeError("Invalid Stripe account ID format")
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

if not ROUTING_TABLE:
    logger.info(
        "No Stripe routing rules configured; installing sandbox placeholder routes"
    )
    ROUTING_TABLE[("stripe", "default", "sandbox", "placeholder")] = {
        "product_id": "prod_sandbox",
        "price_id": "price_sandbox",
        "customer_id": "cus_sandbox",
        "account_id": STRIPE_MASTER_ACCOUNT_ID,
        "secret_key": STRIPE_SECRET_KEY,
        "public_key": STRIPE_PUBLIC_KEY,
    }

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
        acct: Mapping[str, Any] | None = None
        if client:
            # The Stripe client bindings have evolved over time.  Older
            # versions exposed a capitalised ``Account`` resource while newer
            # releases prefer ``accounts`` (lowercase) and in some cases expect
            # an explicit account identifier.  We probe the available
            # attributes dynamically so the router remains compatible across
            # library versions without relying on brittle attribute access.
            account_callables: list[Any] = []
            for attr in ("Account", "account", "accounts"):
                resource = getattr(client, attr, None)
                retrieve = getattr(resource, "retrieve", None)
                if callable(retrieve):
                    account_callables.append(retrieve)
            for retrieve in account_callables:
                try:
                    acct = retrieve()
                except TypeError:
                    # Some retrieve implementations require an explicit
                    # account identifier.  Continue probing other call styles
                    # before falling back to the module-level helper.
                    continue
                if acct is not None:
                    break
            if acct is None:
                if stripe is None:
                    raise RuntimeError("stripe library unavailable")
                acct = stripe.Account.retrieve(api_key=api_key)
        else:
            if stripe is None:
                raise RuntimeError("stripe library unavailable")
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


def iter_master_events(
    *,
    event_types: Iterable[str] | None = None,
    created: Mapping[str, Any] | None = None,
    **params: Any,
):
    """Yield Stripe events for the master account via ``auto_paging_iter``."""

    kwargs: dict[str, Any] = dict(params)
    if event_types is not None:
        kwargs["types"] = list(event_types)
    if created is not None:
        kwargs["created"] = dict(created)
    api_key = STRIPE_SECRET_KEY
    client = _client(api_key)
    try:
        if client:
            events = client.Event.list(**kwargs)
        else:
            events = stripe.Event.list(api_key=api_key, **kwargs)
        return events.auto_paging_iter()
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.exception("Stripe event listing failed: %s", exc)
        raise RuntimeError("Stripe event listing failed") from exc
def _verify_route(bot_id: str, route: Mapping[str, str]) -> None:
    """Validate the resolved route before executing payment actions."""

    if _skip_stripe_verification():
        return

    key = route.get("secret_key")
    allowed = _allowed_secret_keys()
    if not key or key not in allowed:
        _alert_mismatch(bot_id, route.get("account_id") or "unknown")
        raise RuntimeError("Stripe account mismatch")
    account_id = route.get("account_id", STRIPE_MASTER_ACCOUNT_ID)
    if account_id != STRIPE_MASTER_ACCOUNT_ID:
        _alert_mismatch(bot_id, account_id)
        raise RuntimeError("Stripe account mismatch")


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
    route.setdefault("account_id", STRIPE_MASTER_ACCOUNT_ID)
    route.setdefault("currency", "usd")
    for strategy in _STRATEGIES:
        route = strategy.apply(bot_id, dict(route))
    route.setdefault("account_id", STRIPE_MASTER_ACCOUNT_ID)
    route.setdefault("currency", "usd")
    secret = route.get("secret_key", "")
    public = route.get("public_key", "")
    if not secret or not public:
        logger.error("Resolved route missing Stripe keys for bot '%s'", bot_id)
        _alert_mismatch(bot_id, route.get("account_id", ""), "Stripe key misconfiguration")
        raise RuntimeError("Stripe keys are not configured for the resolved route")
    if secret.startswith("sk_test") or public.startswith("pk_test"):
        logger.error("Test mode Stripe API keys are not permitted for bot '%s'", bot_id)
        _alert_mismatch(bot_id, route.get("account_id", ""), "Stripe key misconfiguration")
        raise RuntimeError("Test mode Stripe API keys are not permitted")
    if not secret.startswith("sk_") or not public.startswith("pk_"):
        logger.error("Invalid Stripe API key format for bot '%s'", bot_id)
        _alert_mismatch(bot_id, route.get("account_id", ""), "Stripe key misconfiguration")
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
    _verify_route(bot_id, route)
    api_key = route["secret_key"]
    key_hash = _hash_api_key(api_key)
    client = _client(api_key)

    account_id = _get_account_id(api_key) or ""
    if account_id != STRIPE_MASTER_ACCOUNT_ID or not route.get("secret_key"):
        _alert_mismatch(bot_id, account_id, amount=amount)
        raise RuntimeError("Stripe account mismatch")

    price = price_id or route.get("price_id")
    customer = route.get("customer_id")
    description = description or route.get("product_id", "")
    currency = route.get("currency", "usd")
    user_email = route.get("user_email")
    if customer:
        try:  # pragma: no cover - best effort
            cust_obj = stripe.Customer.retrieve(customer, api_key=api_key)
            if isinstance(cust_obj, Mapping):
                possible_email = cust_obj.get("email")
                if possible_email:
                    user_email = str(possible_email)
        except Exception:
            pass

    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    amt: float | None = None
    had_error = False

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
    except Exception:
        had_error = True
        raise
    finally:
        destination = route.get("account_id") or account_id
        if not destination:
            destination = _get_account_id(api_key)
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
                or destination
            )
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
        event_id = event.get("id") if isinstance(event, Mapping) else None
        email = user_email

        if destination and destination != STRIPE_MASTER_ACCOUNT_ID:
            billing_logger.log_event(
                id=event_id,
                action_type="charge",
                amount=logged_amount,
                currency=currency,
                timestamp_ms=timestamp_ms,
                user_email=email,
                bot_id=bot_id,
                destination_account=destination,
                charge_id=event_id,
                raw_event_json=raw_json,
                error=True,
            )
            _alert_mismatch(bot_id, destination, amount=logged_amount)
            raise RuntimeError("Stripe account mismatch")

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
            id=event_id,
            action_type="charge",
            amount=logged_amount,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            charge_id=event_id,
            raw_event_json=raw_json,
            error=had_error,
        )
        record_payment(
            "charge",
            logged_amount,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
            charge_id=event_id,
        )
        if event is not None:
            log_billing_event(
                "charge",
                bot_id=bot_id,
                amount=logged_amount,
                currency=currency,
                user_email=email,
                destination_account=destination,
                key_hash=key_hash,
                stripe_id=event_id,
                ts=datetime.utcnow().isoformat(),
            )
        _log_payment(
            "charge",
            bot_id,
            amt if amt is not None else 0.0,
            currency,
            email,
            account_id,
            timestamp_ms,
            event_id,
        )
    return event


# Backward compatibility
initiate_charge = charge
init_charge = charge


def get_balance(
    bot_id: str, *, overrides: Optional[Mapping[str, str]] = None
) -> float:
    """Return available balance for the given bot."""
    route = _resolve_route(bot_id, overrides)
    _verify_route(bot_id, route)
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
        raise RuntimeError("Stripe balance retrieval failed") from exc


def create_customer(
    bot_id: str,
    customer_info: Mapping[str, Any],
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Create a new Stripe customer for the given bot."""
    route = _resolve_route(bot_id, overrides)
    _verify_route(bot_id, route)
    api_key = route["secret_key"]
    client = _client(api_key)
    if client:
        event = client.Customer.create(**customer_info)
    else:
        event = stripe.Customer.create(api_key=api_key, **customer_info)
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
    _verify_route(bot_id, route)
    api_key = route["secret_key"]
    key_hash = _hash_api_key(api_key)
    client = _client(api_key)
    account_id = _get_account_id(api_key) or ""
    if account_id != STRIPE_MASTER_ACCOUNT_ID or not route.get("secret_key"):
        _alert_mismatch(bot_id, account_id)
        raise RuntimeError("Stripe account mismatch")
    price = price_id or route.get("price_id")
    customer = customer_id or route.get("customer_id")
    email = route.get("user_email")
    if customer:
        try:  # pragma: no cover - best effort
            cust_obj = stripe.Customer.retrieve(customer, api_key=api_key)
            if isinstance(cust_obj, Mapping):
                possible_email = cust_obj.get("email")
                if possible_email:
                    email = str(possible_email)
        except Exception:
            pass
    if not price or not customer:
        logger.error(
            "price_id and customer_id are required for subscriptions for bot '%s'",
            bot_id,
        )
        raise RuntimeError("price_id and customer_id are required for subscriptions")
    sub_params = {"customer": customer, "items": [{"price": price}], **params}
    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    had_error = False
    try:
        if client:
            event = client.Subscription.create(**sub_params)
        else:
            event = stripe.Subscription.create(api_key=api_key, **sub_params)
    except Exception:
        had_error = True
        raise
    finally:
        currency = route.get("currency")
        destination = route.get("account_id") or account_id
        if not destination:
            destination = _get_account_id(api_key)
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
                or destination
            )
        raw_json = None
        if isinstance(event, Mapping):
            try:
                raw_json = json.dumps(event)
            except Exception:  # pragma: no cover - serialization issues
                raw_json = None
        event_id = event.get("id") if isinstance(event, Mapping) else None

        if destination and destination != STRIPE_MASTER_ACCOUNT_ID:
            billing_logger.log_event(
                id=event_id,
                action_type="subscription",
                amount=None,
                currency=currency,
                timestamp_ms=timestamp_ms,
                user_email=email,
                bot_id=bot_id,
                destination_account=destination,
                charge_id=event_id,
                raw_event_json=raw_json,
                error=True,
            )
            _alert_mismatch(bot_id, destination)
            raise RuntimeError("Stripe account mismatch")

        billing_logger.log_event(
            id=event_id,
            action_type="subscription",
            amount=None,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            charge_id=event_id,
            raw_event_json=raw_json,
            error=had_error,
        )
        record_payment(
            "subscription",
            None,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
            charge_id=event_id,
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
                key_hash=key_hash,
                stripe_id=event_id,
                ts=datetime.utcnow().isoformat(),
            )
        _log_payment(
            "subscription",
            bot_id,
            0.0,
            currency,
            email,
            account_id,
            timestamp_ms,
            event_id,
        )
    return event


def refund(
    bot_id: str,
    payment_intent_id: str,
    *,
    amount: float | None = None,
    overrides: Optional[Mapping[str, str]] = None,
    **params: Any,
) -> dict[str, Any]:
    """Refund a payment for the given bot using ``payment_intent_id``."""

    route = _resolve_route(bot_id, overrides)
    _verify_route(bot_id, route)
    api_key = route["secret_key"]
    key_hash = _hash_api_key(api_key)
    client = _client(api_key)
    account_id = _get_account_id(api_key) or ""
    if account_id != STRIPE_MASTER_ACCOUNT_ID or not route.get("secret_key"):
        _alert_mismatch(bot_id, account_id, amount=amount)
        raise RuntimeError("Stripe account mismatch")
    refund_params: dict[str, Any] = {"payment_intent": payment_intent_id, **params}
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
    customer = route.get("customer_id")
    email = route.get("user_email")
    if customer:
        try:  # pragma: no cover - best effort
            cust_obj = stripe.Customer.retrieve(customer, api_key=api_key)
            if isinstance(cust_obj, Mapping):
                possible_email = cust_obj.get("email")
                if possible_email:
                    email = str(possible_email)
        except Exception:
            pass
    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    had_error = False
    try:
        if client:
            event = client.Refund.create(**refund_params)
        else:
            event = stripe.Refund.create(api_key=api_key, **refund_params)
    except Exception:
        had_error = True
        raise
    finally:
        currency = route.get("currency")
        destination = route.get("account_id") or account_id
        if not destination:
            destination = _get_account_id(api_key)
        logged_amount: float | None = None
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
                or destination
            )
            possible = event.get("amount")
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

        event_id = event.get("id") if isinstance(event, Mapping) else None

        if destination and destination != STRIPE_MASTER_ACCOUNT_ID:
            billing_logger.log_event(
                id=event_id,
                action_type="refund",
                amount=logged_amount,
                currency=currency,
                timestamp_ms=timestamp_ms,
                user_email=email,
                bot_id=bot_id,
                destination_account=destination,
                charge_id=event_id,
                raw_event_json=raw_json,
                error=True,
            )
            _alert_mismatch(bot_id, destination, amount=logged_amount)
            raise RuntimeError("Stripe account mismatch")

        billing_logger.log_event(
            id=event_id,
            action_type="refund",
            amount=logged_amount,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            charge_id=event_id,
            raw_event_json=raw_json,
            error=had_error,
        )
        record_payment(
            "refund",
            logged_amount,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
            charge_id=event_id,
        )
        log_billing_event(
            "refund",
            bot_id=bot_id,
            amount=logged_amount,
            currency=currency,
            user_email=email,
            destination_account=destination,
            key_hash=key_hash,
            stripe_id=event_id,
            ts=datetime.utcnow().isoformat(),
        )
        _log_payment(
            "refund",
            bot_id,
            amount if amount is not None else 0.0,
            currency,
            email,
            account_id,
            timestamp_ms,
            event_id,
        )
    return event


def create_checkout_session(
    bot_id: str,
    line_items: list[Mapping[str, Any]],
    *,
    overrides: Optional[Mapping[str, str]] = None,
    **params: Any,
) -> dict[str, Any]:
    """Create a Stripe Checkout session for the given bot."""

    route = _resolve_route(bot_id, overrides)
    _verify_route(bot_id, route)
    api_key = route["secret_key"]
    key_hash = _hash_api_key(api_key)
    client = _client(api_key)
    account_id = _get_account_id(api_key) or ""
    if account_id != STRIPE_MASTER_ACCOUNT_ID or not route.get("secret_key"):
        _alert_mismatch(bot_id, account_id)
        raise RuntimeError("Stripe account mismatch")
    final_params: dict[str, Any] = {"line_items": list(line_items), **params}
    if "customer" not in final_params and route.get("customer_id"):
        final_params["customer"] = route["customer_id"]
    email = route.get("user_email")
    if final_params.get("customer"):
        try:  # pragma: no cover - best effort
            cust_obj = stripe.Customer.retrieve(final_params["customer"], api_key=api_key)
            if isinstance(cust_obj, Mapping):
                possible_email = cust_obj.get("email")
                if possible_email:
                    email = str(possible_email)
        except Exception:
            pass
    amount_param = final_params.get("amount")
    timestamp_ms = int(time.time() * 1000)
    event: dict[str, Any] | None = None
    had_error = False
    try:
        if client:
            event = client.checkout.Session.create(**final_params)
        else:
            event = stripe.checkout.Session.create(api_key=api_key, **final_params)
    except Exception:
        had_error = True
        raise
    finally:
        currency = route.get("currency")
        destination = route.get("account_id") or account_id
        if not destination:
            destination = _get_account_id(api_key)
        logged_amount: float | None = None
        if isinstance(event, Mapping):
            destination = (
                event.get("on_behalf_of")
                or event.get("account")
                or (event.get("transfer_data") or {}).get("destination")
                or destination
            )
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

        raw_json = None
        if isinstance(event, Mapping):
            try:
                raw_json = json.dumps(event)
            except Exception:  # pragma: no cover - serialization issues
                raw_json = None

        event_id = event.get("id") if isinstance(event, Mapping) else None

        if destination and destination != STRIPE_MASTER_ACCOUNT_ID:
            billing_logger.log_event(
                id=event_id,
                action_type="checkout_session",
                amount=logged_amount,
                currency=currency,
                timestamp_ms=timestamp_ms,
                user_email=email,
                bot_id=bot_id,
                destination_account=destination,
                charge_id=event_id,
                raw_event_json=raw_json,
                error=True,
            )
            _alert_mismatch(bot_id, destination, amount=logged_amount)
            raise RuntimeError("Stripe account mismatch")

        billing_logger.log_event(
            id=event_id,
            action_type="checkout_session",
            amount=logged_amount,
            currency=currency,
            timestamp_ms=timestamp_ms,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            charge_id=event_id,
            raw_event_json=raw_json,
            error=had_error,
        )
        record_payment(
            "checkout_session",
            logged_amount,
            bot_id,
            destination,
            email=email,
            ts=timestamp_ms,
            charge_id=event_id,
        )
        log_billing_event(
            "checkout_session",
            bot_id=bot_id,
            amount=logged_amount,
            currency=currency,
            user_email=email,
            destination_account=destination,
            key_hash=key_hash,
            stripe_id=event_id,
            ts=datetime.utcnow().isoformat(),
        )
        _log_payment(
            "checkout",
            bot_id,
            float(amount_param) if amount_param is not None else 0.0,
            currency,
            email,
            account_id,
            timestamp_ms,
            event_id,
        )
    return event


__all__ = [
    "initiate_charge",
    "init_charge",
    "charge",
    "get_balance",
    "create_customer",
    "create_subscription",
    "refund",
    "create_checkout_session",
    "iter_master_events",
    "register_route",
    "register_override",
    "register_strategy",
    "RouteStrategy",
    "ROUTING_TABLE",
    "BILLING_RULES",
]
