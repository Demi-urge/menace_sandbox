from __future__ import annotations

"""Audit recent Stripe refund and failure events against internal logs."""

import json
import logging
from datetime import datetime, timedelta
from typing import Iterable, Tuple

import stripe_billing_router

from db_router import GLOBAL_ROUTER, init_db_router
from billing.billing_logger import _get_connection as _ledger_connection, log_event
from billing.billing_log_db import BillingLogDB

logger = logging.getLogger(__name__)


def _approved_bot_names() -> set[str]:
    """Return bot names whose status is ``approved``.

    Bots are stored in the ``bots`` table.  We compare against the final segment
    of a Stripe ``bot_id`` (``domain:category:bot`` -> ``bot``).
    """
    router = GLOBAL_ROUTER or init_db_router("default")
    try:
        conn = router.get_connection("bots")
        cur = conn.execute("SELECT name FROM bots WHERE status = ?", ("approved",))
        return {row[0] for row in cur.fetchall()}
    except Exception as exc:  # pragma: no cover - DB failures are logged
        logger.exception("failed fetching approved bot names: %s", exc)
        return set()


def _event_logged(event_id: str, bot_id: str, action: str) -> bool:
    """Return ``True`` if *event_id* already exists in internal logs."""
    conn = _ledger_connection()
    if conn is not None:
        cur = conn.execute("SELECT 1 FROM stripe_ledger WHERE id=?", (event_id,))
        if cur.fetchone():
            return True
    bdb = BillingLogDB()
    cur = bdb.conn.execute(
        "SELECT 1 FROM billing_logs WHERE bot_id=? AND action=?",
        (bot_id, action),
    )
    return cur.fetchone() is not None


def _extract_bot_id(obj: dict) -> str:
    meta = obj.get("metadata") or {}
    return str(meta.get("bot_id") or meta.get("bot") or "")


def _iter_recent_events(hours: int) -> Iterable[object]:
    since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
    return stripe_billing_router.iter_master_events(
        event_types=[
            "charge.refunded",
            "charge.failed",
            "payment_intent.payment_failed",
        ],
        created={"gte": since},
    )


def audit_recent_events(hours: int = 24) -> None:
    """Audit recent Stripe refund/failure events.

    Events for approved bots lacking a matching entry in either
    ``stripe_ledger`` or ``billing_logs`` are logged with ``error=1``.
    """
    approved = _approved_bot_names()
    missing: list[Tuple[object, str, str]] = []
    for event in _iter_recent_events(hours):
        obj = event.data.object.to_dict() if hasattr(event.data.object, "to_dict") else dict(event.data.object)
        bot_id = _extract_bot_id(obj)
        bot_name = bot_id.split(":")[-1]
        if not bot_name or bot_name not in approved:
            continue
        action = "refund" if "refunded" in event.type else "failure"
        if not _event_logged(event.id, bot_id, action):
            logger.warning("Missing workflow log for Stripe event %s (%s)", event.id, action)
            missing.append((event, bot_id, action))

    for event, bot_id, action in missing:
        obj = event.data.object.to_dict() if hasattr(event.data.object, "to_dict") else dict(event.data.object)
        amount = obj.get("amount") or obj.get("amount_refunded")
        currency = obj.get("currency")
        email = obj.get("receipt_email") or obj.get("billing_details", {}).get("email")
        destination = getattr(event, "account", None) or obj.get("on_behalf_of")
        log_event(
            id=event.id,
            action_type=f"unlogged_{action}",
            amount=amount,
            currency=currency,
            timestamp_ms=int(event.created) * 1000,
            user_email=email,
            bot_id=bot_id,
            destination_account=destination,
            raw_event_json=json.dumps(event.to_dict()),
            error=1,
        )


__all__ = ["audit_recent_events"]


if __name__ == "__main__":  # pragma: no cover - CLI usage
    audit_recent_events()
