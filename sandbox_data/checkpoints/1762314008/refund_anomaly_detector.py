from __future__ import annotations

"""Detect anomalies in Stripe refunds and payment failures.

Recent Stripe refund and failed payment intent events are cross checked against
``billing_log_db`` entries and a whitelist of approved Menace workflow or bot
identifiers.  Any refund or failure lacking an authorised initiator or matching
billing log is recorded via :mod:`billing.billing_logger`.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, List

import stripe_billing_router  # noqa: F401 - import required for safety checks
from dynamic_path_router import resolve_path

from billing import billing_logger
from billing.billing_log_db import BillingLogDB
from menace_sanity_layer import record_billing_event, record_payment_anomaly
from typing import TYPE_CHECKING

try:  # Optional dependency – self-coding engine
    from self_coding_engine import SelfCodingEngine  # type: ignore
    from code_database import CodeDB  # type: ignore
    from menace_memory_manager import MenaceMemoryManager  # type: ignore
except Exception:  # pragma: no cover - best effort
    SelfCodingEngine = None  # type: ignore
    CodeDB = None  # type: ignore
    MenaceMemoryManager = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vector_service.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

# Path to JSON file containing approved workflow/bot identifiers
WHITELIST_PATH = resolve_path("billing/approved_workflows.json")
# Default path used for generation parameter tuning updates
CONFIG_PATH = resolve_path("config/stripe_watchdog.yaml")


def load_whitelist(path: Path = WHITELIST_PATH) -> set[str]:
    """Return set of approved workflow or bot identifiers from *path*."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return {str(x) for x in data}
    except FileNotFoundError:
        logger.warning("Whitelist file %s not found", path)
    except Exception:  # pragma: no cover - JSON errors are logged
        logger.exception("Failed to load whitelist", extra={"path": str(path)})
    return set()


def _extract_bot_id(obj: dict) -> str:
    meta = obj.get("metadata") or {}
    return str(meta.get("bot_id") or meta.get("bot") or "")


def _iter_recent_events(hours: int) -> Iterable[object]:
    api_key = os.getenv("STRIPE_API_KEY")
    if not api_key:
        raise RuntimeError("STRIPE_API_KEY not set")
    mod = import_module("stripe")
    mod.api_key = api_key
    since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
    return mod.Event.list(
        types=["charge.refunded", "payment_intent.payment_failed"],
        created={"gte": since},
    ).auto_paging_iter()


def detect_anomalies(
    hours: int = 24,
    *,
    whitelist_path: Path | str = WHITELIST_PATH,
    db_path: Path | str | None = None,
    self_coding_engine: Any | None = None,
    config_path: Path | str | None = CONFIG_PATH,
    context_builder: "ContextBuilder",
) -> List[dict]:
    """Return list of refund or failure anomalies.

    Each anomaly dict contains ``id``, ``bot_id`` and ``reason`` where reason is
    ``'unauthorized'`` or ``'unlogged'``.  Anomalies are also recorded via
    :func:`billing.billing_logger.log_event` with ``error=1``.
    """

    approved = load_whitelist(Path(whitelist_path))
    db = BillingLogDB(db_path) if db_path else BillingLogDB()

    engine = self_coding_engine
    if engine is None and SelfCodingEngine and CodeDB and MenaceMemoryManager:
        try:  # pragma: no cover - best effort
            context_builder.refresh_db_weights()
            engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=context_builder
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to initialise SelfCodingEngine")

    anomalies: List[dict] = []
    for event in _iter_recent_events(hours):
        obj = event.data.object
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        elif not isinstance(obj, dict):
            obj = obj.__dict__
        bot_id = _extract_bot_id(obj)
        action = "refund" if "refunded" in event.type else "failure"

        amount = obj.get("amount") or obj.get("amount_refunded")
        currency = obj.get("currency")
        email = obj.get("receipt_email") or obj.get("billing_details", {}).get("email")
        destination = getattr(event, "account", None) or obj.get("on_behalf_of")
        event_dict = (
            event.to_dict() if hasattr(event, "to_dict") else {"id": event.id, "type": event.type}
        )
        raw_json = json.dumps(event_dict)

        if bot_id not in approved:
            billing_logger.log_event(
                id=event.id,
                action_type=f"unauthorized_{action}",
                amount=amount,
                currency=currency,
                timestamp_ms=int(getattr(event, "created", 0)) * 1000,
                user_email=email,
                bot_id=bot_id,
                destination_account=destination,
                charge_id=event.id,
                raw_event_json=raw_json,
                error=1,
            )
            record_billing_event(
                "refund_anomaly" if action == "refund" else "payment_failure",
                {
                    "stripe_event_id": event.id,
                    "stripe_object_id": obj.get("id"),
                    "bot_id": bot_id,
                    "reason": "unauthorized",
                },
                (
                    f"{action.capitalize()} event {event.id} for bot {bot_id} was unauthorized; "
                    "ensure all Stripe payments are authorized and logged."
                ),
                config_path=config_path,
                self_coding_engine=engine,
            )
            anomalies.append(
                {"id": event.id, "bot_id": bot_id, "reason": "unauthorized"}
            )
            record_payment_anomaly(
                f"unauthorized_{action}",
                {
                    "stripe_event_id": event.id,
                    "stripe_object_id": obj.get("id"),
                    "bot_id": bot_id,
                    "action": action,
                },
                (
                    f"{action.capitalize()} event {event.id} for bot {bot_id} was unauthorized; "
                    "ensure all Stripe payments are authorized and logged."
                ),
                self_coding_engine=engine,
            )
            continue

        event_time = datetime.utcfromtimestamp(getattr(event, "created", 0))
        start_ts = (event_time - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        end_ts = (event_time + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        cur = db.conn.execute(
            "SELECT 1 FROM billing_logs WHERE bot_id=? AND action=? AND ts>=? AND ts<=?",
            (bot_id, action, start_ts, end_ts),
        )
        if cur.fetchone() is None:
            billing_logger.log_event(
                id=event.id,
                action_type=f"unlogged_{action}",
                amount=amount,
                currency=currency,
                timestamp_ms=int(getattr(event, "created", 0)) * 1000,
                user_email=email,
                bot_id=bot_id,
                destination_account=destination,
                charge_id=event.id,
                raw_event_json=raw_json,
                error=1,
            )
            record_billing_event(
                "refund_anomaly" if action == "refund" else "payment_failure",
                {
                    "stripe_event_id": event.id,
                    "stripe_object_id": obj.get("id"),
                    "bot_id": bot_id,
                    "reason": "unlogged",
                },
                (
                    f"{action.capitalize()} event {event.id} for bot {bot_id} was unlogged; "
                    "ensure all Stripe payments are authorized and logged."
                ),
                config_path=config_path,
                self_coding_engine=engine,
            )
            anomalies.append({"id": event.id, "bot_id": bot_id, "reason": "unlogged"})
            record_payment_anomaly(
                "missing_refund" if action == "refund" else "missing_failure_log",
                {
                    "stripe_event_id": event.id,
                    "stripe_object_id": obj.get("id"),
                    "bot_id": bot_id,
                    "action": action,
                },
                (
                    f"{action.capitalize()} event {event.id} for bot {bot_id} was unlogged; "
                    "ensure all Stripe payments are authorized and logged."
                ),
                self_coding_engine=engine,
            )

    return anomalies


__all__ = ["detect_anomalies", "load_whitelist"]
