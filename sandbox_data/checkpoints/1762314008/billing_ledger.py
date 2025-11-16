from __future__ import annotations

"""Lightweight billing ledger for recording payment actions.

This module writes JSON lines entries to ``finance_logs/stripe_ledger.jsonl``.
It exposes :func:`log_action` for low level logging and :func:`record_payment`
for convenience when recording payment related events from the Stripe billing
router.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - optional dependency during bootstrap
    import stripe_billing_router  # noqa: F401
except Exception:  # pragma: no cover - best effort import
    stripe_billing_router = None  # type: ignore

try:  # resolve path dynamically when available
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - optional dependency
    resolve_path = None  # type: ignore

# Determine ledger file path and ensure directory exists
if resolve_path is not None:
    try:
        _log_dir = resolve_path("finance_logs")
    except FileNotFoundError:
        _log_dir = Path("finance_logs")
else:
    _log_dir = Path("finance_logs")

_log_dir.mkdir(parents=True, exist_ok=True)
_LEDGER_FILE = _log_dir / "stripe_ledger.jsonl"

# Global lock to ensure thread safe appends
_LOCK = threading.Lock()


def log_action(
    action: str,
    amount: Optional[float],
    bot_id: str,
    account_id: str,
    email: Optional[str] = None,
    ts: Optional[int] = None,
    charge_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """Append a log entry describing a billing ``action``.

    Parameters mirror the required fields with ``extra`` allowing future
    expansion without breaking the interface.  ``ts`` defaults to the current
    time in milliseconds.
    """

    record = {
        "ts": int(ts if ts is not None else time.time() * 1000),
        "action": action,
        "amount": amount,
        "bot_id": bot_id,
        "account_id": account_id,
        "email": email,
        "charge_id": charge_id,
    }
    record.update(extra)

    line = json.dumps(record, sort_keys=True)
    with _LOCK:  # ensure atomic append
        with _LEDGER_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def record_payment(
    action: str,
    amount: Optional[float],
    bot_id: str,
    account_id: str,
    *,
    email: Optional[str] = None,
    ts: Optional[int] = None,
    charge_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """Public helper to record a payment related ``action``."""

    log_action(
        action,
        amount,
        bot_id,
        account_id,
        email=email,
        ts=ts,
        charge_id=charge_id,
        **extra,
    )


__all__ = ["log_action", "record_payment"]
