from __future__ import annotations

"""Stripe anomaly detection watchdog.

This module cross checks recent Stripe activity against local billing logs
and ROI projections.  It can be executed periodically by ``cron`` to flag
inconsistencies.

Key features implemented according to the specification:

* The Stripe API key is loaded from :class:`VaultSecretProvider` when
  available or the ``STRIPE_SECRET_KEY`` environment variable.  If neither is
  configured the key falls back to the global key configured on the ``stripe``
  client (if present).
* Helper functions fetch recent charges, refunds and events using the Stripe
  API and read local ledger records produced by ``billing_logger`` or
  ``StripeLedger``.
* Detection routines flag:
    - charges missing from the local ledger
    - refunds or failed payments lacking Menace workflow logs
    - unexpected webhook endpoints
    - revenue mismatches compared to ROI projections
* Anomalies are logged via :func:`audit_logger.log_event` using the
  ``stripe_anomaly`` event type.  When ``--write-codex`` is supplied the
  anomaly is also emitted as a ``TrainingSample`` for downstream Codex
  ingestion.
* :func:`main` exposes a CLI so the watchdog can be run by ``cron``.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import audit_logger
import yaml
import alert_dispatcher
from dynamic_path_router import resolve_path

try:  # Optional dependency – Stripe API client
    import stripe  # type: ignore
except Exception:  # pragma: no cover - best effort
    stripe = None  # type: ignore

try:  # Optional dependency – secrets from the vault
    from vault_secret_provider import VaultSecretProvider  # type: ignore
except Exception:  # pragma: no cover - best effort
    VaultSecretProvider = None  # type: ignore

try:  # Optional dependency – central Stripe routing config
    from stripe_billing_router import STRIPE_SECRET_KEY  # type: ignore
except Exception:  # pragma: no cover - best effort
    STRIPE_SECRET_KEY = None  # type: ignore

try:  # Optional dependency – Codex training sample helper
    from codex_db_helpers import TrainingSample  # type: ignore
except Exception:  # pragma: no cover - best effort
    TrainingSample = None  # type: ignore

try:  # Optional dependency – ROI projections
    from roi_results_db import ROIResultsDB  # type: ignore
except Exception:  # pragma: no cover - best effort
    ROIResultsDB = None  # type: ignore

try:  # Optional dependency – structured billing ledger
    from billing.stripe_ledger import STRIPE_LEDGER  # type: ignore
except Exception:  # pragma: no cover - best effort
    STRIPE_LEDGER = None  # type: ignore

try:  # Optional dependency – billing log database
    from billing.billing_log_db import BillingLogDB  # type: ignore
except Exception:  # pragma: no cover - best effort
    BillingLogDB = None  # type: ignore

try:  # Optional dependency – discrepancy logging
    from discrepancy_db import DiscrepancyDB, DiscrepancyRecord  # type: ignore
except Exception:  # pragma: no cover - best effort
    DiscrepancyDB = None  # type: ignore
    DiscrepancyRecord = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default set of allowed webhook IDs or URLs. Any endpoint returned by Stripe
#: outside this set (and the configured list) is flagged as an anomaly.
DEFAULT_ALLOWED_WEBHOOKS = {
    "https://menace.example.com/stripe/webhook",
}

#: Default log file for anomaly summaries.
ANOMALY_LOG = resolve_path("stripe_watchdog.log")

#: Fallback path used when ``StripeLedger`` is unavailable
LEDGER_FILE = resolve_path("finance_logs/stripe_ledger.jsonl")

#: Path to YAML configuration containing the list of allowed webhook
#: endpoints.
CONFIG_PATH = resolve_path("config/stripe_watchdog.yaml")


def _load_allowed_webhooks(path: Path | None = None) -> set[str]:
    """Return allowed webhook identifiers from environment or ``path``."""

    allowed = set(DEFAULT_ALLOWED_WEBHOOKS)

    env_val = os.getenv("STRIPE_ALLOWED_WEBHOOKS")
    if env_val:
        allowed.update(x.strip() for x in env_val.split(",") if x.strip())
        return allowed

    cfg = path or CONFIG_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return allowed
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg})
        return allowed

    endpoints = data.get("allowed_webhooks") or data.get("authorized_webhooks")
    if isinstance(endpoints, list):
        allowed.update(str(url) for url in endpoints if isinstance(url, str))
    return allowed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter(obj: Iterable | object) -> Iterable:
    """Iterate over ``obj`` respecting Stripe's auto paging interface."""

    pager = getattr(obj, "auto_paging_iter", None)
    return pager() if callable(pager) else obj  # type: ignore[return-value]


# API key loading -----------------------------------------------------------

def load_api_key() -> Optional[str]:
    """Return the Stripe API key.

    The key is sourced from :mod:`stripe_billing_router`'s
    :data:`STRIPE_SECRET_KEY` constant or via ``VaultSecretProvider`` when
    available.  This avoids relying on environment variables so the watchdog
    can run in more restricted environments.
    """

    key = STRIPE_SECRET_KEY
    if not key and VaultSecretProvider:
        try:  # pragma: no cover - best effort
            key = VaultSecretProvider().get("stripe_secret_key")
        except Exception:
            key = None
    if not key:
        logger.error("Stripe API key not configured")
    return key


# Stripe fetchers -----------------------------------------------------------


def fetch_recent_charges(api_key: str, start_ts: int, end_ts: int) -> List[dict]:
    """Return Stripe charges created within ``start_ts``..``end_ts``."""

    if stripe is None:
        return []
    try:  # pragma: no cover - network request
        charges = stripe.Charge.list(
            api_key=api_key,
            limit=100,
            created={"gte": int(start_ts), "lt": int(end_ts)},
        )
    except TypeError:  # older stubs used in tests may not accept "created"
        try:
            charges = stripe.Charge.list(limit=100, api_key=api_key)
        except Exception:  # pragma: no cover - best effort
            logger.exception("Stripe charge fetch failed")
            return []
    except Exception:  # pragma: no cover - best effort
        logger.exception("Stripe charge fetch failed")
        return []
    return [dict(c) if isinstance(c, dict) else c.to_dict_recursive() for c in _iter(charges)]


def fetch_recent_refunds(api_key: str, start_ts: int, end_ts: int) -> List[dict]:
    """Return Stripe refunds created within ``start_ts``..``end_ts``."""

    if stripe is None:
        return []
    try:  # pragma: no cover - network request
        refunds = stripe.Refund.list(
            api_key=api_key,
            limit=100,
            created={"gte": int(start_ts), "lt": int(end_ts)},
        )
    except TypeError:  # fallback for stub implementations
        try:
            refunds = stripe.Refund.list(limit=100, api_key=api_key)
        except Exception:  # pragma: no cover - best effort
            logger.exception("Stripe refund fetch failed")
            return []
    except Exception:  # pragma: no cover - best effort
        logger.exception("Stripe refund fetch failed")
        return []
    return [dict(r) if isinstance(r, dict) else r.to_dict_recursive() for r in _iter(refunds)]


def fetch_recent_events(api_key: str, start_ts: int, end_ts: int) -> List[dict]:
    """Return Stripe events for failed payments or refunds."""

    if stripe is None:
        return []
    types = ["charge.failed", "payment_intent.payment_failed", "charge.refunded"]
    try:  # pragma: no cover - network request
        events = stripe.Event.list(
            api_key=api_key,
            limit=100,
            created={"gte": int(start_ts), "lt": int(end_ts)},
            types=types,
        )
    except TypeError:  # fallback for stub implementations
        try:
            events = stripe.Event.list(limit=100, api_key=api_key)
        except Exception:  # pragma: no cover - best effort
            logger.exception("Stripe event fetch failed")
            return []
    except Exception:  # pragma: no cover - best effort
        logger.exception("Stripe event fetch failed")
        return []
    return [dict(e) if isinstance(e, dict) else e.to_dict_recursive() for e in _iter(events)]


# Local ledger --------------------------------------------------------------


def load_local_ledger(start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
    """Return ledger rows between ``start_ts`` and ``end_ts``.

    The function first attempts to query :class:`StripeLedger` when available
    and falls back to the JSONL ledger produced by ``billing_logger``.
    ``start_ts``/``end_ts`` are Unix timestamps in seconds.
    """

    start_ms, end_ms = int(start_ts * 1000), int(end_ts * 1000)
    rows: List[Dict[str, Any]] = []

    # Read from the JSONL ledger when available (used in tests)
    path = LEDGER_FILE
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("timestamp_ms")
                    if isinstance(ts, (int, float)) and start_ms <= int(ts) < end_ms:
                        rows.append(rec)
        except Exception:  # pragma: no cover - best effort
            logger.exception("Failed to read local ledger")
        return rows

    # Fallback to the structured ledger database
    if STRIPE_LEDGER is not None:  # pragma: no branch - optional path
        try:
            conn = STRIPE_LEDGER.router.get_connection("stripe_ledger")
            cur = conn.execute(
                "SELECT id, action, amount, timestamp FROM stripe_ledger "
                "WHERE timestamp >= ? AND timestamp < ?",
                (start_ms, end_ms),
            )
            for cid, action, amount, ts in cur.fetchall():
                rows.append(
                    {
                        "id": str(cid),
                        "action_type": str(action),
                        "amount": float(amount),
                        "timestamp_ms": int(ts),
                    }
                )
        except Exception:  # pragma: no cover - best effort
            logger.exception("StripeLedger query failed")
        return rows

    return rows


# Billing logs --------------------------------------------------------------


def load_billing_logs(start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
    """Return ``billing_logs`` rows between ``start_ts`` and ``end_ts``."""

    if BillingLogDB is None:
        return []
    start_iso = datetime.utcfromtimestamp(start_ts).isoformat()
    end_iso = datetime.utcfromtimestamp(end_ts).isoformat()
    rows: List[Dict[str, Any]] = []
    try:  # pragma: no cover - best effort
        db = BillingLogDB()
        cur = db.conn.execute(
            "SELECT amount, ts FROM billing_logs WHERE action = ? AND ts >= ? AND ts < ?",
            ("charge", start_iso, end_iso),
        )
        for amount, ts in cur.fetchall():
            try:
                ts_epoch = datetime.fromisoformat(ts).timestamp() if ts else None
            except Exception:
                ts_epoch = None
            rows.append(
                {
                    "amount": float(amount) if amount is not None else None,
                    "timestamp": ts_epoch,
                }
            )
    except Exception:
        logger.exception("BillingLogDB query failed")
    return rows


# Anomaly logging -----------------------------------------------------------


def _emit_anomaly(record: Dict[str, Any], write_codex: bool) -> None:
    """Log *record* and optionally emit a Codex training sample."""

    audit_logger.log_event("stripe_anomaly", record)
    try:
        with ANOMALY_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        logger.exception("Failed to write anomaly log", extra={"record": record})
    if write_codex and TrainingSample is not None:
        try:  # pragma: no cover - best effort
            TrainingSample(source="stripe_watchdog", content=json.dumps(record))
        except Exception:
            logger.exception("Failed to create Codex training sample")


# Detection routines -------------------------------------------------------


def detect_missing_charges(
    charges: Iterable[dict],
    ledger: List[Dict[str, Any]],
    billing_logs: List[Dict[str, Any]] | None = None,
    *,
    write_codex: bool = False,
) -> List[Dict[str, Any]]:
    """Return Stripe charges absent from local billing logs."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("id")}
    ledger_ts = {
        int(e.get("timestamp_ms"))
        for e in ledger
        if isinstance(e.get("timestamp_ms"), (int, float))
    }
    billing_index: Dict[float, List[int]] = {}
    if billing_logs:
        for rec in billing_logs:
            amt = rec.get("amount")
            ts = rec.get("timestamp")
            if isinstance(amt, (int, float)) and isinstance(ts, (int, float)):
                billing_index.setdefault(round(float(amt), 2), []).append(int(ts))

    anomalies: List[Dict[str, Any]] = []
    for charge in charges:
        cid = str(charge.get("id"))
        created = charge.get("created")
        created_sec = int(created) if isinstance(created, (int, float)) else None
        created_ms = int(created_sec * 1000) if created_sec is not None else None
        if cid in ledger_ids or (created_ms is not None and created_ms in ledger_ts):
            continue
        if billing_index and created_sec is not None:
            amt = charge.get("amount")
            amount_dollars = (
                round(float(amt) / 100.0, 2) if isinstance(amt, (int, float)) else None
            )
            if amount_dollars is not None:
                ts_list = billing_index.get(amount_dollars, [])
                if any(abs(created_sec - ts) <= 300 for ts in ts_list):
                    continue
        anomaly = {
            "id": cid,
            "amount": charge.get("amount"),
            "email": charge.get("receipt_email"),
            "timestamp": created,
        }
        anomalies.append(anomaly)
        _emit_anomaly(anomaly, write_codex)
    return anomalies


def detect_missing_refunds(
    refunds: Iterable[dict],
    ledger: List[Dict[str, Any]],
    *,
    write_codex: bool = False,
) -> List[Dict[str, Any]]:
    """Return Stripe refunds not present in ``ledger``."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("action_type") == "refund"}
    anomalies: List[Dict[str, Any]] = []
    for refund in refunds:
        rid = str(refund.get("id"))
        if rid in ledger_ids:
            continue
        anomaly = {
            "type": "missing_refund",
            "refund_id": rid,
            "amount": refund.get("amount"),
            "charge": refund.get("charge"),
        }
        anomalies.append(anomaly)
        _emit_anomaly(anomaly, write_codex)
    return anomalies


def detect_failed_events(
    events: Iterable[dict],
    ledger: List[Dict[str, Any]],
    *,
    write_codex: bool = False,
) -> List[Dict[str, Any]]:
    """Return failed payment events missing from the ledger."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("action_type") == "failed"}
    anomalies: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") not in {"charge.failed", "payment_intent.payment_failed"}:
            continue
        eid = str(event.get("id"))
        if eid in ledger_ids:
            continue
        anomaly = {
            "type": "missing_failure_log",
            "event_id": eid,
            "event_type": event.get("type"),
        }
        anomalies.append(anomaly)
        _emit_anomaly(anomaly, write_codex)
    return anomalies


def check_webhook_endpoints(
    api_key: str,
    approved: Iterable[str] | None = None,
    *,
    write_codex: bool = False,
) -> List[str]:
    """Return webhook identifiers failing the allowed or enabled checks."""

    if stripe is None:
        return []
    try:  # pragma: no cover - network request
        endpoints = stripe.WebhookEndpoint.list(api_key=api_key)
    except Exception:  # pragma: no cover - best effort
        logger.exception("Stripe webhook endpoint listing failed")
        return []

    allowed = set(str(u) for u in (approved or _load_allowed_webhooks()))
    flagged: List[str] = []
    for ep in _iter(endpoints):
        if isinstance(ep, dict):
            ep_dict = ep
        else:
            try:
                ep_dict = ep.to_dict_recursive()
            except Exception:  # pragma: no cover - best effort
                ep_dict = {k: getattr(ep, k, None) for k in ("id", "url", "status")}

        ep_id = ep_dict.get("id")
        url = ep_dict.get("url")
        status = ep_dict.get("status")
        identifier = ep_id or url
        if identifier is None:
            continue

        if str(identifier) not in allowed or status != "enabled":
            flagged.append(str(identifier))
            anomaly_type = "disabled_webhook" if status != "enabled" else "unknown_webhook"
            record = {"type": anomaly_type, "id": ep_id, "url": url, "status": status}
            _emit_anomaly(record, write_codex)
            try:  # pragma: no cover - best effort
                alert_type = (
                    "stripe_disabled_endpoint"
                    if anomaly_type == "disabled_webhook"
                    else "stripe_unknown_endpoint"
                )
                alert_dispatcher.dispatch_alert(
                    alert_type,
                    3,
                    f"Stripe webhook issue: {identifier}",
                    record,
                )
            except Exception:
                logger.exception(
                    "alert dispatch failed", extra={"id": identifier, "status": status}
                )
    return flagged


def _projected_revenue_between(start_ts: int, end_ts: int) -> float:
    """Return projected revenue logged between ``start_ts`` and ``end_ts``."""

    if ROIResultsDB is None:
        return 0.0
    try:  # pragma: no cover - DB access
        db = ROIResultsDB()
        method = getattr(db, "projected_revenue_between", None)
        if callable(method):
            return float(method(start_ts, end_ts))
        start_iso = datetime.utcfromtimestamp(start_ts).isoformat()
        end_iso = datetime.utcfromtimestamp(end_ts).isoformat()
        cur = db.conn.cursor()
        cur.execute(
            "SELECT SUM(roi_gain) FROM workflow_results "
            "WHERE timestamp >= ? AND timestamp < ?",
            (start_iso, end_iso),
        )
        row = cur.fetchone()
        return float(row[0] or 0.0)
    except Exception:  # pragma: no cover - best effort
        logger.exception("ROIResultsDB query failed")
        return 0.0


def compare_revenue(
    charges: Iterable[dict],
    refunds: Iterable[dict],
    *,
    tolerance: float = 0.1,
    write_codex: bool = False,
) -> Optional[Dict[str, float]]:
    """Compare Stripe net revenue with projected revenue from ROI logs."""

    total = 0.0
    for ch in charges:
        if ch.get("status") == "succeeded" and isinstance(ch.get("amount"), (int, float)):
            total += float(ch["amount"])
    for rf in refunds:
        amt = rf.get("amount")
        if isinstance(amt, (int, float)):
            total -= float(amt)
    net_revenue = total / 100.0

    projected = 0.0
    if ROIResultsDB is not None:
        try:  # pragma: no cover - DB access
            projected = ROIResultsDB().projected_revenue()
        except Exception:  # pragma: no cover - best effort
            logger.exception("ROIResultsDB query failed")

    if projected and abs(net_revenue - projected) > tolerance * projected:
        details = {
            "net_revenue": net_revenue,
            "projected_revenue": projected,
            "difference": net_revenue - projected,
        }
        record = {"type": "revenue_mismatch", **details}
        _emit_anomaly(record, write_codex)
        try:  # pragma: no cover - best effort
            alert_dispatcher.dispatch_alert(
                "stripe_revenue_mismatch",
                4,
                json.dumps(details),
                details,
            )
        except Exception:
            logger.exception("alert dispatch failed", extra=details)
        return details
    return None


def compare_revenue_window(
    start_ts: int,
    end_ts: int,
    *,
    tolerance: float = 0.1,
    write_codex: bool = False,
) -> Optional[Dict[str, float]]:
    """Compare Stripe revenue and ROI projections for a time window."""

    api_key = load_api_key()
    if not api_key:
        return None
    charges = fetch_recent_charges(api_key, start_ts, end_ts)
    refunds = fetch_recent_refunds(api_key, start_ts, end_ts)

    total = 0.0
    for ch in charges:
        if ch.get("status") == "succeeded" and isinstance(ch.get("amount"), (int, float)):
            total += float(ch["amount"])
    for rf in refunds:
        amt = rf.get("amount")
        if isinstance(amt, (int, float)):
            total -= float(amt)
    net_revenue = total / 100.0

    projected = _projected_revenue_between(start_ts, end_ts)

    if projected and abs(net_revenue - projected) > tolerance * projected:
        details = {
            "net_revenue": net_revenue,
            "projected_revenue": projected,
            "difference": net_revenue - projected,
        }
        record = {
            "type": "revenue_mismatch",
            "start_ts": start_ts,
            "end_ts": end_ts,
            **details,
        }
        _emit_anomaly(record, write_codex)
        try:  # pragma: no cover - best effort
            alert_dispatcher.dispatch_alert(
                "stripe_revenue_mismatch",
                4,
                json.dumps(details),
                details,
            )
        except Exception:
            logger.exception("alert dispatch failed", extra=details)
        return details
    return None


# Convenience wrappers used by tests and the CLI ---------------------------


def check_events(hours: int = 1, *, write_codex: bool = False) -> List[Dict[str, Any]]:
    """Check for missing charges within the last ``hours``."""

    api_key = load_api_key()
    if not api_key:
        return []
    end_ts = int(time.time())
    start_ts = end_ts - int(hours * 3600)
    check_webhook_endpoints(api_key, write_codex=write_codex)
    # Load the entire ledger window; many tests use historical timestamps.
    ledger = load_local_ledger(0, end_ts)
    billing_logs = load_billing_logs(0, end_ts)
    charges = fetch_recent_charges(api_key, start_ts, end_ts)
    anomalies = detect_missing_charges(
        charges, ledger, billing_logs, write_codex=write_codex
    )
    if anomalies and DiscrepancyDB and DiscrepancyRecord:
        try:  # pragma: no cover - best effort
            msg = f"{len(anomalies)} stripe anomalies detected"
            DiscrepancyDB().add(
                DiscrepancyRecord(message=msg, metadata={"count": len(anomalies)})
            )
        except Exception:
            logger.exception("Failed to record discrepancy summary")
    return anomalies


def check_revenue_projection(
    hours: int = 1, *, tolerance: float = 0.1, write_codex: bool = False
) -> Optional[Dict[str, float]]:
    """Compare revenue for the last ``hours`` against projections."""
    end_ts = int(time.time())
    start_ts = end_ts - int(hours * 3600)
    return compare_revenue_window(
        start_ts, end_ts, tolerance=tolerance, write_codex=write_codex
    )


# CLI ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for command line execution."""

    global ANOMALY_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=str,
        help="ISO timestamp or epoch seconds for start of window",
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default=str(ANOMALY_LOG),
        help="path to anomaly audit log",
    )
    parser.add_argument(
        "--write-codex",
        action="store_true",
        help="also emit Codex training samples",
    )
    args = parser.parse_args(argv)

    api_key = load_api_key()
    if not api_key:
        logger.error("Cannot run watchdog without Stripe API key")
        return

    end_ts = int(time.time())
    if args.since:
        try:
            start_ts = int(datetime.fromisoformat(args.since).timestamp())
        except ValueError:
            try:
                start_ts = int(float(args.since))
            except ValueError:
                logger.error("Invalid --since value: %s", args.since)
                return
    else:
        start_ts = end_ts - 3600

    ANOMALY_LOG = Path(args.audit_log)
    ledger = load_local_ledger(start_ts, end_ts)
    billing_logs = load_billing_logs(start_ts, end_ts)

    charges = fetch_recent_charges(api_key, start_ts, end_ts)
    refunds = fetch_recent_refunds(api_key, start_ts, end_ts)
    events = fetch_recent_events(api_key, start_ts, end_ts)

    detect_missing_charges(charges, ledger, billing_logs, write_codex=args.write_codex)
    detect_missing_refunds(refunds, ledger, write_codex=args.write_codex)
    detect_failed_events(events, ledger, write_codex=args.write_codex)
    check_webhook_endpoints(api_key, write_codex=args.write_codex)
    compare_revenue_window(start_ts, end_ts, write_codex=args.write_codex)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    logging.basicConfig(level=logging.INFO)
    main()
