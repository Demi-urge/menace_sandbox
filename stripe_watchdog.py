from __future__ import annotations

"""Cross-check recent Stripe charges against the local ledger."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List

import yaml
from dynamic_path_router import resolve_path

import alert_dispatcher

try:  # pragma: no cover - optional dependency
    from vault_secret_provider import VaultSecretProvider
except Exception:  # pragma: no cover - best effort
    VaultSecretProvider = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import stripe_billing_router as sbr
except Exception:  # pragma: no cover - best effort
    sbr = None  # type: ignore
    stripe = None  # type: ignore
else:  # pragma: no cover - optional dependency
    stripe = getattr(sbr, "stripe", None)

logger = logging.getLogger(__name__)


def _ledger_path() -> Path:
    try:
        log_dir = resolve_path("finance_logs")
    except FileNotFoundError:  # pragma: no cover - fallback
        log_dir = Path("finance_logs")
    return log_dir / "stripe_ledger.jsonl"


LEDGER_FILE = _ledger_path()

CONFIG_PATH = resolve_path("config/stripe_watchdog.yaml")


def _load_allowed_endpoints(path: Path | None = None) -> set[str]:
    """Return configured set of allowed webhook endpoint URLs."""

    cfg_path = path or CONFIG_PATH
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Stripe watchdog config missing", extra={"path": cfg_path})
        return set()
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg_path})
        return set()

    endpoints = data.get("allowed_endpoints")
    if not isinstance(endpoints, list):
        logger.warning("allowed_endpoints not configured correctly", extra={"path": cfg_path})
        return set()
    return {str(url) for url in endpoints if isinstance(url, str)}


def check_webhook_endpoints(api_key: str | None = None) -> List[str]:
    """Alert if Stripe webhook endpoints differ from configuration."""

    api_key = api_key or load_api_key()
    if not api_key or stripe is None:
        return []

    allowed = _load_allowed_endpoints()
    try:  # pragma: no cover - network issues
        endpoints = stripe.WebhookEndpoint.list(api_key=api_key)
    except Exception:
        logger.exception("Stripe API request for webhook endpoints failed")
        return []

    unknown: List[str] = []
    for ep in _iter(endpoints):
        url = getattr(ep, "url", None)
        if url is None and isinstance(ep, dict):
            url = ep.get("url")
        if url and url not in allowed:
            unknown.append(url)

    if unknown:
        msg = f"Unknown Stripe webhook endpoints: {unknown}"
        logger.error(msg)
        try:  # pragma: no cover - best effort
            alert_dispatcher.dispatch_alert("stripe_unknown_endpoint", 5, msg)
        except Exception:
            logger.exception("alert dispatch failed for webhook check")
    else:
        logger.info("All Stripe webhook endpoints accounted for")
    return unknown


def load_api_key() -> str | None:
    """Return the Stripe API key from env or the secret vault."""

    provider = VaultSecretProvider() if VaultSecretProvider else None
    env_name = "STRIPE_" + "SECRET_KEY"
    key = os.getenv(env_name)
    if not key and provider is not None:
        key = provider.get("stripe_secret_key")
    if not key:
        logger.error("Stripe API key not configured")
    return key


def _ledger_entries() -> List[dict[str, Any]]:
    """Return all records from the billing ledger."""

    entries: List[dict[str, Any]] = []
    if not LEDGER_FILE.exists():
        return entries
    try:
        with LEDGER_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    entries.append(rec)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed reading %s", LEDGER_FILE)
    return entries


def _iter(obj: Iterable | object) -> Iterable:
    pager = getattr(obj, "auto_paging_iter", None)
    if callable(pager):
        return pager()
    return obj  # type: ignore[return-value]


def _charge_email(charge: Any) -> str | None:
    """Return email associated with ``charge`` if available."""

    if isinstance(charge, dict):
        email = charge.get("receipt_email")
        if email:
            return email
        bd = charge.get("billing_details")
        if isinstance(bd, dict):
            return bd.get("email")
        return None

    email = getattr(charge, "receipt_email", None)
    if email:
        return email
    bd = getattr(charge, "billing_details", None)
    if isinstance(bd, dict):
        return bd.get("email")
    return getattr(bd, "email", None)


def check_events() -> List[dict[str, Any]]:
    """Return anomalies for Stripe charges missing from the billing logs."""

    api_key = load_api_key()
    if not api_key or stripe is None:
        return []

    # Also verify registered webhook endpoints
    check_webhook_endpoints(api_key)

    try:
        charges = stripe.Charge.list(limit=100, api_key=api_key)
    except Exception:  # pragma: no cover - network issues
        logger.exception("Stripe API request failed")
        return []

    entries = _ledger_entries()
    ids = {e.get("id") for e in entries if e.get("id")}
    timestamps = {e.get("timestamp_ms") for e in entries if e.get("timestamp_ms")}

    anomalies: List[dict[str, Any]] = []
    for charge in _iter(charges):
        cid = getattr(charge, "id", None)
        if cid is None and isinstance(charge, dict):
            cid = charge.get("id")
        created = getattr(charge, "created", None)
        if created is None and isinstance(charge, dict):
            created = charge.get("created")
        created_ms = int(created * 1000) if isinstance(created, (int, float)) else None
        if cid in ids or (created_ms and created_ms in timestamps):
            continue
        amount = getattr(charge, "amount", None)
        if amount is None and isinstance(charge, dict):
            amount = charge.get("amount")
        anomaly = {
            "id": cid,
            "amount": amount,
            "email": _charge_email(charge),
            "timestamp": created,
        }
        anomalies.append(anomaly)

    if anomalies:
        logger.warning("Charges missing from billing logs: %s", anomalies)
    else:
        logger.info("All Stripe charges logged")
    return anomalies


def main() -> None:
    """Run the watchdog hourly using APScheduler if available."""

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except Exception:  # pragma: no cover - fallback
        logger.exception("APScheduler unavailable, running once")
        check_events()
        return

    scheduler = BlockingScheduler()
    scheduler.add_job(check_events, "interval", hours=1)
    scheduler.start()


if __name__ == "__main__":
    main()
