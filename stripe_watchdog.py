from __future__ import annotations

"""Cross-check Stripe charges and refunds against the local ledger."""

import json
import logging
import os
from pathlib import Path
from typing import Iterable

from dynamic_path_router import resolve_path

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


def _ledger_ids() -> set[str]:
    ids: set[str] = set()
    if not LEDGER_FILE.exists():
        return ids
    try:
        with LEDGER_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id"):
                    ids.add(rec["id"])
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed reading %s", LEDGER_FILE)
    return ids


def _iter(obj: Iterable | object) -> Iterable:
    pager = getattr(obj, "auto_paging_iter", None)
    if callable(pager):
        return pager()
    return obj  # type: ignore[return-value]


def check_events() -> list[str]:
    """Return IDs of Stripe events missing from the local ledger."""

    api_key = load_api_key()
    if not api_key or stripe is None:
        return []

    try:
        charges = stripe.Charge.list(limit=100, api_key=api_key)
        refunds = stripe.Refund.list(limit=100, api_key=api_key)
    except Exception:  # pragma: no cover - network issues
        logger.exception("Stripe API request failed")
        return []

    logged = _ledger_ids()
    missing: list[str] = []

    for charge in _iter(charges):
        cid = getattr(charge, "id", None)
        if not cid and isinstance(charge, dict):
            cid = charge.get("id")
        if cid and cid not in logged:
            missing.append(cid)

    for refund in _iter(refunds):
        rid = getattr(refund, "id", None)
        if not rid and isinstance(refund, dict):
            rid = refund.get("id")
        if rid and rid not in logged:
            missing.append(rid)

    if missing:
        logger.warning("Missing Stripe events: %s", ", ".join(missing))
    else:
        logger.info("All Stripe events logged")
    return missing


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
