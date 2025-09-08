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
* Allowed webhook endpoints, anomaly hints and severities are sourced from
  YAML configuration files so updates can take effect without code changes.
* Recent feedback snippets from :func:`menace_sanity_layer.fetch_recent_billing_issues`
  are consulted before detection to lower severities or suppress acknowledged
  false positives.  Set ``adaptive_issue_handling`` to ``false`` in
  ``config/stripe_watchdog.yaml`` to disable this adaptive behaviour.
* :func:`main` exposes a CLI so the watchdog can be run by ``cron``.
* When ``menace_sanity_layer`` is unavailable, stub functions log a single
  warning then raise :class:`SanityLayerUnavailableError` on subsequent use.
  Set the environment variable ``MENACE_SANITY_OPTIONAL`` to bypass the raise
  (a critical alert is logged instead).
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import audit_logger
import yaml
import alert_dispatcher
from dynamic_path_router import resolve_path
from audit_trail import AuditTrail
from logging.handlers import RotatingFileHandler
import gzip
import shutil
import types

logger = logging.getLogger(__name__)

try:  # pragma: no cover - best effort to import sanity layer
    import menace_sanity_layer
    from menace_sanity_layer import (
        record_billing_anomaly,
        record_billing_event,
        record_event,
    )
except Exception:  # pragma: no cover - fallback stubs if import fails
    logger.warning(
        "menace_sanity_layer import failed; using no-op stubs for feedback",
    )
    menace_sanity_layer = types.SimpleNamespace(
        record_payment_anomaly=lambda *a, **k: None,
        EVENT_TYPE_INSTRUCTIONS={},
        refresh_billing_instructions=lambda *a, **k: None,
        fetch_recent_billing_issues=lambda *a, **k: [],
    )

    _SANITY_LAYER_OPTIONAL = os.getenv("MENACE_SANITY_OPTIONAL")

    class SanityLayerUnavailableError(RuntimeError):
        """Raised when ``menace_sanity_layer`` is required but missing."""

    def _escalate(msg: str) -> None:
        """Raise or log a critical alert depending on configuration."""
        if _SANITY_LAYER_OPTIONAL:
            logger.critical(msg)
        else:
            raise SanityLayerUnavailableError(msg)

    def record_billing_anomaly(*_a, **_k):  # noqa: D401 - simple stub
        """Fallback stub when menace_sanity_layer is unavailable."""
        if not getattr(record_billing_anomaly, "_warned", False):
            logger.warning(
                "record_billing_anomaly stub invoked; menace_sanity_layer missing",
            )
            record_billing_anomaly._warned = True
        else:
            _escalate("record_billing_anomaly stub called without menace_sanity_layer")

    def record_billing_event(*_a, **_k):  # noqa: D401 - simple stub
        """Fallback stub when menace_sanity_layer is unavailable."""
        if not getattr(record_billing_event, "_warned", False):
            logger.warning(
                "record_billing_event stub invoked; menace_sanity_layer missing",
            )
            record_billing_event._warned = True
        else:
            _escalate("record_billing_event stub called without menace_sanity_layer")

    def record_event(*_a, **_k):  # noqa: D401 - simple stub
        """Fallback stub when menace_sanity_layer is unavailable."""
        if not getattr(record_event, "_warned", False):
            logger.warning(
                "record_event stub invoked; menace_sanity_layer missing",
            )
            record_event._warned = True
        else:
            _escalate("record_event stub called without menace_sanity_layer")

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

try:  # Optional dependency – self-coding feedback
    from self_coding_engine import SelfCodingEngine  # type: ignore
    try:
        from vector_service.context_builder import ContextBuilder  # type: ignore
    except Exception:  # pragma: no cover - fallback when context builder missing
        ContextBuilder = None  # type: ignore
    from code_database import CodeDB  # type: ignore
    from menace_memory_manager import MenaceMemoryManager  # type: ignore
except Exception:  # pragma: no cover - best effort
    SelfCodingEngine = None  # type: ignore
    ContextBuilder = None  # type: ignore
    CodeDB = None  # type: ignore
    MenaceMemoryManager = None  # type: ignore

try:  # Optional dependency – telemetry feedback loop
    from telemetry_feedback import TelemetryFeedback  # type: ignore
    from error_logger import ErrorLogger  # type: ignore
except Exception:  # pragma: no cover - best effort
    TelemetryFeedback = None  # type: ignore
    ErrorLogger = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Fallback set of allowed webhook IDs or URLs. Any endpoint returned by Stripe
#: outside the configured list is flagged as an anomaly when it doesn't appear
#: in the YAML configuration or ``STRIPE_ALLOWED_WEBHOOKS`` environment variable.
_DEFAULT_ALLOWED_WEBHOOKS = {
    "https://menace.example.com/stripe/webhook",
}

#: Default per-anomaly severities used when logging events to the sanity layer.
#: These values can be overridden via ``config/billing_instructions.yaml`` which
#: is monitored by :func:`_refresh_instruction_cache`.
DEFAULT_SEVERITY_MAP = {
    "missing_charge": 2.5,
    "missing_refund": 2.0,
    "missing_failure_log": 1.5,
    "unapproved_workflow": 3.5,
    "unknown_webhook": 2.0,
    "disabled_webhook": 3.0,
    "revenue_mismatch": 4.0,
    "account_mismatch": 3.0,
    "unauthorized_charge": 3.5,
    "unauthorized_refund": 3.5,
    "unauthorized_failure": 3.0,
}
SEVERITY_MAP = DEFAULT_SEVERITY_MAP.copy()

# Module identifiers used for targeted remediation by downstream consumers.
BILLING_ROUTER_MODULE = "stripe_billing_router"
WATCHDOG_MODULE = "stripe_watchdog"

#: Default log file for anomaly summaries.
_LOG_DIR = resolve_path("finance_logs")
#: JSON lines file used for anomaly audit records.
ANOMALY_LOG = _LOG_DIR / "stripe_watchdog_audit.jsonl"
#: Marker storing the last successful run timestamp.
_LAST_RUN_FILE = _LOG_DIR / "stripe_watchdog_last_run.txt"
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5MB
_DEFAULT_BACKUP_COUNT = 5
#: Path to YAML configuration containing the list of allowed webhook endpoints
CONFIG_PATH = resolve_path("config/stripe_watchdog.yaml")
#: Path used when exporting normalized anomalies for training purposes.
TRAINING_EXPORT = resolve_path("training_data/stripe_anomalies.jsonl")

# Instruction and severity overrides for sanity layer feedback. ``menace_sanity_layer``
# caches the file contents so we track the modification time and refresh when it
# changes.
_BILLING_INSTRUCTIONS_PATH = Path(resolve_path("config/billing_instructions.yaml"))
_BILLING_INSTRUCTIONS_MTIME = 0.0


def _load_severity_map(path: Path | None = None) -> Dict[str, float]:
    """Return per-anomaly severities from ``path`` or defaults."""

    cfg = path or _BILLING_INSTRUCTIONS_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        sev = data.get("severity_map", {})
        if isinstance(sev, dict):
            return {
                str(k): float(v)
                for k, v in sev.items()
                if isinstance(v, (int, float))
            }
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception(
            "Failed to load billing instructions", extra={"path": str(cfg)}
        )
    return DEFAULT_SEVERITY_MAP.copy()


def _refresh_instruction_cache() -> None:
    """Reload billing instructions and severity map when the config file changes."""

    global _BILLING_INSTRUCTIONS_MTIME, SEVERITY_MAP
    try:
        mtime = _BILLING_INSTRUCTIONS_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    if mtime != _BILLING_INSTRUCTIONS_MTIME:
        menace_sanity_layer.refresh_billing_instructions(_BILLING_INSTRUCTIONS_PATH)
        SEVERITY_MAP = _load_severity_map(_BILLING_INSTRUCTIONS_PATH)
        _BILLING_INSTRUCTIONS_MTIME = mtime


def _sanity_feedback_enabled(path: Path | None = None) -> bool:
    """Return whether Sanity Layer feedback is enabled in config."""

    cfg = path or CONFIG_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return bool(data.get("sanity_layer_feedback", True))
    except FileNotFoundError:
        return True
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg})
        return True


SANITY_LAYER_FEEDBACK_ENABLED = _sanity_feedback_enabled()


def _adaptive_issue_handling_enabled(path: Path | None = None) -> bool:
    """Return whether adaptive issue handling is enabled."""

    cfg = path or CONFIG_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return bool(data.get("adaptive_issue_handling", True))
    except FileNotFoundError:
        return True
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg})
        return True


ADAPTIVE_ISSUE_HANDLING_ENABLED = _adaptive_issue_handling_enabled()
SKIP_ANOMALY_TYPES: set[str] = set()

# Load billing instructions and severity map at module import.
_refresh_instruction_cache()

# Generic instruction used when no specific guidance exists for an event type.
DEFAULT_BILLING_EVENT_INSTRUCTION = (
    "Avoid generating bots that issue Stripe charges without logging through billing_logger."
)

# Backwards compatible alias for older callers/tests expecting
# ``BILLING_EVENT_INSTRUCTION``.
BILLING_EVENT_INSTRUCTION = DEFAULT_BILLING_EVENT_INSTRUCTION


def _load_log_rotation(path: Path | None = None) -> tuple[int, int]:
    """Return ``maxBytes`` and ``backupCount`` from ``path`` or defaults."""

    cfg = path or CONFIG_PATH
    max_bytes = _DEFAULT_MAX_BYTES
    backup_count = _DEFAULT_BACKUP_COUNT
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        rotation = data.get("log_rotation") or {}
        max_bytes = int(rotation.get("maxBytes", rotation.get("max_bytes", max_bytes)))
        backup_count = int(
            rotation.get("backupCount", rotation.get("backup_count", backup_count))
        )
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg})
    return max_bytes, backup_count


_ISSUE_LOWER_RE = re.compile(r"lower severity (?P<atype>[\w_]+)", re.I)
_ISSUE_IGNORE_RE = re.compile(r"(?:ignore|skip) (?P<atype>[\w_]+)", re.I)
_ACK_MISMATCH_RE = re.compile(r"acknowledged mismatch", re.I)


def _parse_issue_snippets(
    snippets: Iterable[str],
) -> tuple[Dict[str, float], set[str]]:
    """Return severity overrides and anomaly types to skip."""

    overrides: Dict[str, float] = {}
    skip: set[str] = set()
    for sn in snippets:
        s = sn.strip().lower()
        match = _ISSUE_LOWER_RE.search(s)
        if match:
            overrides[match.group("atype")] = 0.5
        match = _ISSUE_IGNORE_RE.search(s)
        if match:
            skip.add(match.group("atype"))
        if _ACK_MISMATCH_RE.search(s):
            overrides.setdefault("account_mismatch", 0.5)
    return overrides, skip


def _fetch_and_apply_recent_issues() -> None:
    """Retrieve recent billing issues and adapt severity/skip lists."""

    if not ADAPTIVE_ISSUE_HANDLING_ENABLED:
        return
    try:
        snippets = menace_sanity_layer.fetch_recent_billing_issues()
    except Exception:
        logger.exception("Failed to fetch recent billing issues")
        return
    overrides, skip = _parse_issue_snippets(snippets)
    for atype, factor in overrides.items():
        base = SEVERITY_MAP.get(atype, DEFAULT_SEVERITY_MAP.get(atype, 1.0))
        SEVERITY_MAP[atype] = base * factor
    SKIP_ANOMALY_TYPES.update(skip)


def _gzip_rotator(source: str, dest: str) -> None:
    with open(source, "rb") as sf, gzip.open(dest, "wb") as df:
        shutil.copyfileobj(sf, df)
    os.remove(source)


def _prepare_anomaly_log() -> RotatingFileHandler:
    """Ensure log directory exists and return a rotating log handler."""

    ANOMALY_LOG.parent.mkdir(parents=True, exist_ok=True)
    max_bytes, backup_count = _load_log_rotation()
    handler = RotatingFileHandler(
        str(ANOMALY_LOG),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.rotator = _gzip_rotator
    handler.namer = lambda name: name + ".gz"
    return handler


ANOMALY_HANDLER = _prepare_anomaly_log()
ANOMALY_TRAIL = AuditTrail(str(ANOMALY_LOG), handler=ANOMALY_HANDLER)


def _read_last_run_ts() -> int:
    """Return the timestamp of the last successful watchdog run."""

    try:
        return int(_LAST_RUN_FILE.read_text().strip())
    except Exception:
        return 0


def _write_last_run_ts(ts: int) -> None:
    """Persist the timestamp of the latest watchdog run."""

    try:
        _LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_RUN_FILE.write_text(str(int(ts)))
    except Exception:  # pragma: no cover - best effort
        logger.exception("Failed to update last-run marker")


#: Fallback path used when ``StripeLedger`` is unavailable
LEDGER_FILE = resolve_path("finance_logs/stripe_ledger.jsonl")


def _load_allowed_webhooks(path: Path | None = None) -> set[str]:
    """Return allowed webhook identifiers from environment or YAML config."""

    cfg = path or CONFIG_PATH
    allowed: set[str] = set()
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        endpoints = data.get("allowed_webhooks") or data.get("authorized_webhooks")
        if isinstance(endpoints, list):
            allowed.update(str(url) for url in endpoints if isinstance(url, str))
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg})

    env_val = os.getenv("STRIPE_ALLOWED_WEBHOOKS")
    if env_val:
        allowed.update(x.strip() for x in env_val.split(",") if x.strip())

    if not allowed:
        allowed.update(_DEFAULT_ALLOWED_WEBHOOKS)

    return allowed


def load_approved_workflows(path: Path | None = None) -> set[str]:
    """Return approved workflow or bot identifiers.

    Reads from the ``STRIPE_APPROVED_WORKFLOWS`` environment variable when
    set or from the YAML config at ``path``/``CONFIG_PATH`` otherwise.
    """

    env_val = os.getenv("STRIPE_APPROVED_WORKFLOWS")
    if env_val:
        return {w.strip() for w in env_val.split(",") if w.strip()}

    cfg = path or CONFIG_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return set()
    except Exception:
        logger.exception(
            "Failed to load Stripe watchdog config", extra={"path": cfg}
        )
        return set()

    workflows = data.get("approved_workflows") or data.get("approved_bot_ids")
    if isinstance(workflows, list):
        return {str(w) for w in workflows if isinstance(w, str)}
    return set()


def _expected_account_id(api_key: str, path: Path | None = None) -> Optional[str]:
    """Return the expected Stripe account identifier.

    The ID is sourced from ``path``/``CONFIG_PATH`` when present; otherwise
    ``stripe.Account.retrieve`` is invoked using ``api_key``.  Any errors result
    in ``None`` being returned.
    """

    cfg = path or CONFIG_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        account_id = (
            data.get("expected_account_id")
            or data.get("stripe_account_id")
            or data.get("account_id")
        )
        if account_id:
            return str(account_id)
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg})
    if stripe is not None:
        try:  # pragma: no cover - network disabled in tests
            acct = stripe.Account.retrieve(api_key=api_key)
            if isinstance(acct, dict):
                return acct.get("id")
            return getattr(acct, "id", None)
        except Exception:
            logger.exception("failed to fetch Stripe account identifier")
    return None


def _allowed_account_ids(path: Path | None = None) -> set[str]:
    """Return the set of allowed Stripe account identifiers from config."""

    cfg = path or CONFIG_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        accounts = (
            data.get("allowed_accounts")
            or data.get("allowed_account_ids")
            or data.get("allowed_account")
            or []
        )
        if isinstance(accounts, (str, int)):
            accounts = [accounts]
        return {str(a) for a in accounts if a}
    except FileNotFoundError:
        return set()
    except Exception:
        logger.exception("Failed to load Stripe watchdog config", extra={"path": cfg})
        return set()

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
    :data:`STRIPE_SECRET_KEY` constant, the ``STRIPE_SECRET_KEY`` environment
    variable or via ``VaultSecretProvider`` when available.  When retrieved the
    key is also assigned to ``stripe.api_key``.
    """

    key = STRIPE_SECRET_KEY or os.getenv("STRIPE_SECRET_KEY")
    if not key and VaultSecretProvider:
        try:  # pragma: no cover - best effort
            key = VaultSecretProvider().get("stripe_secret_key")
        except Exception:
            key = None
    if key and stripe is not None:
        try:  # pragma: no cover - best effort
            stripe.api_key = key
        except Exception:
            logger.exception("Failed to set Stripe API key")
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


def load_billing_logs(
    start_ts: int, end_ts: int, action: str = "charge"
) -> List[Dict[str, Any]]:
    """Return ``billing_logs`` rows for ``action`` between ``start_ts`` and ``end_ts``."""

    if BillingLogDB is None:
        return []
    start_iso = datetime.utcfromtimestamp(start_ts).isoformat()
    end_iso = datetime.utcfromtimestamp(end_ts).isoformat()
    rows: List[Dict[str, Any]] = []
    try:  # pragma: no cover - best effort
        db = BillingLogDB()
        cur = db.conn.execute(
            (
                "SELECT amount, ts, stripe_id, bot_id FROM billing_logs "
                "WHERE action = ? AND ts >= ? AND ts < ?"
            ),
            (action, start_iso, end_iso),
        )
        for amount, ts, sid, bot in cur.fetchall():
            try:
                ts_epoch = datetime.fromisoformat(ts).timestamp() if ts else None
            except Exception:
                ts_epoch = None
            rows.append(
                {
                    "amount": float(amount) if amount is not None else None,
                    "timestamp": ts_epoch,
                    "stripe_id": str(sid) if sid is not None else None,
                    "bot_id": str(bot) if bot is not None else None,
                }
            )
    except Exception:
        logger.exception("BillingLogDB query failed")
    return rows


# Anomaly logging -----------------------------------------------------------


def _emit_anomaly(
    record: Dict[str, Any],
    write_codex: bool,
    export_training: bool,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> None:
    """Log *record* and optionally emit a training sample."""
    _refresh_instruction_cache()
    if record.get("type") in SKIP_ANOMALY_TYPES:
        logger.debug("Skipping anomaly due to adaptive hints", extra=record)
        return
    audit_logger.log_event("stripe_anomaly", record)
    metadata = {k: v for k, v in record.items() if k != "type"}
    metadata.setdefault("timestamp", datetime.utcnow().isoformat())
    if not metadata.get("stripe_account"):
        acct = metadata.get("account_id")
        if not acct:
            try:
                api_key = load_api_key()
                acct = _expected_account_id(api_key) if api_key else None
            except Exception:
                logger.exception(
                    "Failed to resolve Stripe account", extra={"record": record}
                )
        if acct:
            metadata["stripe_account"] = acct

    if SANITY_LAYER_FEEDBACK_ENABLED:
        try:
            record_event(
                record.get("type", "unknown"),
                metadata,
                self_coding_engine=self_coding_engine,
                telemetry_feedback=telemetry_feedback,
            )
        except Exception:
            logger.exception("Failed to record sanity event", extra={"record": record})

    try:
        entry = {
            "type": record.get("type", "unknown"),
            "metadata": metadata,
            "timestamp": int(time.time()),
        }
        ANOMALY_TRAIL.record(entry)
    except Exception:
        logger.exception("Failed to write anomaly log", extra={"record": record})
    if export_training:
        sample = {
            "source": "stripe_watchdog",
            "content": json.dumps(entry),
            "timestamp": entry["timestamp"],
        }
        try:  # pragma: no cover - best effort
            TRAINING_EXPORT.parent.mkdir(parents=True, exist_ok=True)
            with open(TRAINING_EXPORT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample) + "\n")
        except Exception:
            logger.exception("Failed to export anomaly", extra={"record": record})
    if write_codex and TrainingSample is not None:
        try:  # pragma: no cover - best effort
            TrainingSample(source="stripe_watchdog", content=json.dumps(record))
        except Exception:
            logger.exception("Failed to create Codex training sample")
    if SANITY_LAYER_FEEDBACK_ENABLED:
        try:
            severity = SEVERITY_MAP.get(record.get("type"), 1.0)
            record_billing_anomaly(
                record.get("type", "unknown"), record, severity=severity
            )
        except Exception:
            logger.exception("Failed to record billing anomaly", extra={"record": record})

        event_type = record.get("type", "unknown")
        instruction = menace_sanity_layer.EVENT_TYPE_INSTRUCTIONS.get(
            event_type, DEFAULT_BILLING_EVENT_INSTRUCTION
        )
        try:
            record_billing_event(
                event_type,
                metadata,
                instruction,
                config_path=CONFIG_PATH,
                self_coding_engine=self_coding_engine,
            )
        except Exception:
            logger.exception(
                "Failed to record billing event", extra={"record": record}
            )

        payment_meta = dict(metadata)
        if "id" in payment_meta:
            payment_meta["charge_id"] = payment_meta.pop("id")
        payment_meta.setdefault("reason", record.get("type"))
        menace_sanity_layer.record_payment_anomaly(
            event_type,
            payment_meta,
            instruction,
            severity=SEVERITY_MAP.get(event_type, 1.0),
            write_codex=write_codex,
            export_training=export_training,
            self_coding_engine=self_coding_engine,
            telemetry_feedback=telemetry_feedback,
        )


# Detection routines -------------------------------------------------------


def detect_account_mismatches(
    charges: Iterable[dict] | None,
    refunds: Iterable[dict] | None,
    events: Iterable[dict] | None,
    allowed_accounts: Iterable[str],
    *,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return Stripe objects whose account is not in ``allowed_accounts``."""

    allowed = {str(a) for a in allowed_accounts if a}
    anomalies: List[Dict[str, Any]] = []

    def _check(obj: dict, obj_id: str) -> None:
        acct = obj.get("account")
        if acct and allowed and str(acct) not in allowed:
            anomaly = {
                "type": "account_mismatch",
                "id": obj_id,
                "account_id": acct,
                "allowed_accounts": sorted(allowed),
                "module": BILLING_ROUTER_MODULE,
            }
            anomalies.append(anomaly)
            _emit_anomaly(
                anomaly,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )

    for ch in charges or []:
        cid = ch.get("id")
        if cid:
            _check(ch, str(cid))
    for rf in refunds or []:
        rid = rf.get("id")
        if rid:
            _check(rf, str(rid))
    for ev in events or []:
        eid = ev.get("id")
        if eid:
            _check(ev, str(eid))
    return anomalies


def detect_unauthorized_charges(
    charges: Iterable[dict],
    ledger: List[Dict[str, Any]],
    billing_logs: List[Dict[str, Any]] | None = None,
    approved_workflows: Iterable[str] | None = None,
    *,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return charges that bypassed central routing or lack approval."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("id")}
    billing_map = {
        str(rec.get("stripe_id")): rec
        for rec in (billing_logs or [])
        if rec.get("stripe_id")
    }
    anomalies: List[Dict[str, Any]] = []
    for charge in charges:
        cid = str(charge.get("id"))
        if cid in ledger_ids:
            continue
        log = billing_map.get(cid)
        if not log:
            continue
        bot_id = log.get("bot_id")
        anomaly = {
            "type": "unauthorized_charge",
            "charge_id": cid,
            "amount": charge.get("amount"),
            "bot_id": bot_id,
            "account_id": charge.get("account"),
            "module": BILLING_ROUTER_MODULE,
        }
        anomalies.append(anomaly)
        _emit_anomaly(
            anomaly,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
    return anomalies


def detect_unauthorized_refunds(
    refunds: Iterable[dict],
    ledger: List[Dict[str, Any]],
    billing_logs: List[Dict[str, Any]] | None = None,
    approved_workflows: Iterable[str] | None = None,
    *,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return refunds that bypassed central routing or lack approval."""

    ledger_ids = {
        str(e.get("id")) for e in ledger if e.get("action_type") == "refund"
    }
    billing_map = {
        str(rec.get("stripe_id")): rec
        for rec in (billing_logs or [])
        if rec.get("stripe_id")
    }
    approved = {str(w) for w in (approved_workflows or [])}

    anomalies: List[Dict[str, Any]] = []
    for refund in refunds:
        rid = str(refund.get("id"))
        log = billing_map.get(rid)
        if not log:
            continue
        bot_id = log.get("bot_id")
        unauthorized = rid not in ledger_ids or (
            bot_id and approved and bot_id not in approved
        )
        if not unauthorized:
            continue
        anomaly = {
            "type": "unauthorized_refund",
            "refund_id": rid,
            "amount": refund.get("amount"),
            "charge": refund.get("charge"),
            "bot_id": bot_id,
            "account_id": refund.get("account"),
            "module": BILLING_ROUTER_MODULE,
        }
        anomalies.append(anomaly)
        _emit_anomaly(
            anomaly,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
    return anomalies


def detect_missing_charges(
    charges: Iterable[dict],
    ledger: List[Dict[str, Any]],
    billing_logs: List[Dict[str, Any]] | None = None,
    approved_workflows: Iterable[str] | None = None,
    *,
    expected_account_id: str | None = None,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return Stripe charges absent from local billing logs."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("id")}
    billing_map = {
        str(rec.get("stripe_id")): rec
        for rec in (billing_logs or [])
        if rec.get("stripe_id")
    }
    approved = {str(w) for w in (approved_workflows or [])}

    anomalies: List[Dict[str, Any]] = []
    for charge in charges:
        cid = str(charge.get("id"))
        acct = charge.get("account")
        if (
            expected_account_id
            and acct
            and str(acct) != str(expected_account_id)
        ):
            anomaly = {
                "type": "account_mismatch",
                "id": cid,
                "account_id": acct,
                "expected_account_id": expected_account_id,
                "module": BILLING_ROUTER_MODULE,
            }
            anomalies.append(anomaly)
            _emit_anomaly(
                anomaly,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )
        log = billing_map.get(cid)
        bot_id = log.get("bot_id") if log else None
        if bot_id and approved and bot_id not in approved:
            anomaly = {
                "type": "unapproved_workflow",
                "id": cid,
                "bot_id": bot_id,
                "account_id": charge.get("account"),
                "module": BILLING_ROUTER_MODULE,
            }
            anomalies.append(anomaly)
            _emit_anomaly(
                anomaly,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )
            try:  # pragma: no cover - best effort
                alert_dispatcher.dispatch_alert(
                    "stripe_unapproved_workflow",
                    3,
                    f"Unapproved workflow: {bot_id}",
                    anomaly,
                )
            except Exception:
                logger.exception(
                    "alert dispatch failed", extra={"id": cid, "bot_id": bot_id}
                )
        if cid in ledger_ids or log:
            continue
        anomaly = {
            "type": "missing_charge",
            "id": cid,
            "amount": charge.get("amount"),
            "email": charge.get("receipt_email"),
            "timestamp": charge.get("created"),
            "account_id": charge.get("account"),
            "module": BILLING_ROUTER_MODULE,
        }
        anomalies.append(anomaly)
        _emit_anomaly(
            anomaly,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
    return anomalies


def detect_missing_refunds(
    refunds: Iterable[dict],
    ledger: List[Dict[str, Any]],
    billing_logs: List[Dict[str, Any]] | None = None,
    approved_workflows: Iterable[str] | None = None,
    *,
    expected_account_id: str | None = None,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return Stripe refunds absent from the ledger and billing logs."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("action_type") == "refund"}
    billing_map = {
        str(rec.get("stripe_id")): rec
        for rec in (billing_logs or [])
        if rec.get("stripe_id")
    }
    approved = {str(w) for w in (approved_workflows or [])}

    anomalies: List[Dict[str, Any]] = []
    for refund in refunds:
        rid = str(refund.get("id"))
        acct = refund.get("account")
        if (
            expected_account_id
            and acct
            and str(acct) != str(expected_account_id)
        ):
            anomaly = {
                "type": "account_mismatch",
                "refund_id": rid,
                "account_id": acct,
                "expected_account_id": expected_account_id,
                "module": BILLING_ROUTER_MODULE,
            }
            anomalies.append(anomaly)
            _emit_anomaly(
                anomaly,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )
        log = billing_map.get(rid)
        bot_id = log.get("bot_id") if log else None
        if bot_id and approved and bot_id not in approved:
            anomaly = {
                "type": "unapproved_workflow",
                "refund_id": rid,
                "bot_id": bot_id,
                "account_id": refund.get("account"),
                "module": BILLING_ROUTER_MODULE,
            }
            anomalies.append(anomaly)
            _emit_anomaly(
                anomaly,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )
            try:  # pragma: no cover - best effort
                alert_dispatcher.dispatch_alert(
                    "stripe_unapproved_workflow",
                    3,
                    f"Unapproved workflow: {bot_id}",
                    anomaly,
                )
            except Exception:
                logger.exception(
                    "alert dispatch failed", extra={"id": rid, "bot_id": bot_id}
                )
        if rid in ledger_ids or log:
            continue
        anomaly = {
            "type": "missing_refund",
            "refund_id": rid,
            "amount": refund.get("amount"),
            "charge": refund.get("charge"),
            "account_id": refund.get("account"),
            "module": BILLING_ROUTER_MODULE,
        }
        anomalies.append(anomaly)
        _emit_anomaly(
            anomaly,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
    return anomalies


def detect_unauthorized_failures(
    events: Iterable[dict],
    ledger: List[Dict[str, Any]],
    billing_logs: List[Dict[str, Any]] | None = None,
    approved_workflows: Iterable[str] | None = None,
    *,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return failed events present in logs but missing authorization."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("action_type") == "failed"}
    billing_map = {
        str(rec.get("stripe_id")): rec
        for rec in (billing_logs or [])
        if rec.get("stripe_id")
    }
    approved = {str(w) for w in (approved_workflows or [])}

    anomalies: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") not in {"charge.failed", "payment_intent.payment_failed"}:
            continue
        eid = str(event.get("id"))
        log = billing_map.get(eid)
        if not log:
            continue
        bot_id = log.get("bot_id")
        unauthorized = eid not in ledger_ids or (
            bot_id and approved and bot_id not in approved
        )
        if not unauthorized:
            continue
        anomaly = {
            "type": "unauthorized_failure",
            "event_id": eid,
            "event_type": event.get("type"),
            "bot_id": bot_id,
            "account_id": event.get("account"),
            "module": BILLING_ROUTER_MODULE,
        }
        anomalies.append(anomaly)
        _emit_anomaly(
            anomaly,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
    return anomalies


def detect_failed_events(
    events: Iterable[dict],
    ledger: List[Dict[str, Any]],
    billing_logs: List[Dict[str, Any]] | None = None,
    approved_workflows: Iterable[str] | None = None,
    *,
    expected_account_id: str | None = None,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return failed payment events missing from the ledger and billing logs."""

    ledger_ids = {str(e.get("id")) for e in ledger if e.get("action_type") == "failed"}
    billing_map = {
        str(rec.get("stripe_id")): rec
        for rec in (billing_logs or [])
        if rec.get("stripe_id")
    }
    approved = {str(w) for w in (approved_workflows or [])}

    anomalies: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") not in {"charge.failed", "payment_intent.payment_failed"}:
            continue
        eid = str(event.get("id"))
        acct = event.get("account")
        if (
            expected_account_id
            and acct
            and str(acct) != str(expected_account_id)
        ):
            anomaly = {
                "type": "account_mismatch",
                "event_id": eid,
                "account_id": acct,
                "expected_account_id": expected_account_id,
                "module": BILLING_ROUTER_MODULE,
            }
            anomalies.append(anomaly)
            _emit_anomaly(
                anomaly,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )
        log = billing_map.get(eid)
        bot_id = log.get("bot_id") if log else None
        if bot_id and approved and bot_id not in approved:
            anomaly = {
                "type": "unapproved_workflow",
                "event_id": eid,
                "bot_id": bot_id,
                "account_id": event.get("account"),
                "module": BILLING_ROUTER_MODULE,
            }
            anomalies.append(anomaly)
            _emit_anomaly(
                anomaly,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )
            try:  # pragma: no cover - best effort
                alert_dispatcher.dispatch_alert(
                    "stripe_unapproved_workflow",
                    3,
                    f"Unapproved workflow: {bot_id}",
                    anomaly,
                )
            except Exception:
                logger.exception(
                    "alert dispatch failed", extra={"id": eid, "bot_id": bot_id}
                )
        if eid in ledger_ids or log:
            continue
        anomaly = {
            "type": "missing_failure_log",
            "event_id": eid,
            "event_type": event.get("type"),
            "account_id": event.get("account"),
            "module": BILLING_ROUTER_MODULE,
        }
        anomalies.append(anomaly)
        _emit_anomaly(
            anomaly,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
    return anomalies


def check_webhook_endpoints(
    api_key: str,
    approved: Iterable[str] | None = None,
    *,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
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
            record = {
                "type": anomaly_type,
                "webhook_id": ep_id,
                "webhook_url": url,
                "status": status,
                "account_id": ep_dict.get("account"),
                "module": WATCHDOG_MODULE,
            }
            _emit_anomaly(
                record,
                write_codex,
                export_training,
                self_coding_engine,
                telemetry_feedback,
            )
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
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> Optional[Dict[str, float]]:
    """Compare Stripe net revenue with projected revenue from ROI logs."""

    charges = list(charges)
    refunds = list(refunds)
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
        charge_ids = [ch.get("id") for ch in charges if ch.get("id")]
        refund_ids = [rf.get("id") for rf in refunds if rf.get("id")]
        account_ids = list(
            {ch.get("account") for ch in charges if ch.get("account")}
            | {rf.get("account") for rf in refunds if rf.get("account")}
        )
        details = {
            "net_revenue": net_revenue,
            "projected_revenue": projected,
            "difference": net_revenue - projected,
            "charge_ids": charge_ids,
            "refund_ids": refund_ids,
            "account_ids": account_ids,
        }
        record = {"type": "revenue_mismatch", **details}
        _emit_anomaly(
            record,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
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


def summarize_revenue_window(
    start_ts: int,
    end_ts: int,
    *,
    tolerance: float = 0.1,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> Dict[str, float]:
    """Summarize Stripe revenue for ``start_ts``..``end_ts`` and compare to projections.

    Returns a dictionary with charge, refund and projected totals.  When the
    absolute difference between net and projected revenue exceeds ``tolerance``
    times the projection an anomaly is logged via :func:`_emit_anomaly`.
    """

    api_key = load_api_key()
    if not api_key:
        return {
            "charge_total": 0.0,
            "refund_total": 0.0,
            "net_revenue": 0.0,
            "projected_revenue": 0.0,
            "difference": 0.0,
        }

    charges = fetch_recent_charges(api_key, start_ts, end_ts)
    refunds = fetch_recent_refunds(api_key, start_ts, end_ts)

    charge_cents = sum(
        float(ch["amount"])
        for ch in charges
        if ch.get("status") == "succeeded" and isinstance(ch.get("amount"), (int, float))
    )
    refund_cents = sum(
        float(rf.get("amount", 0.0))
        for rf in refunds
        if isinstance(rf.get("amount"), (int, float))
    )

    charge_total = charge_cents / 100.0
    refund_total = refund_cents / 100.0
    net_revenue = (charge_cents - refund_cents) / 100.0

    projected = _projected_revenue_between(start_ts, end_ts)
    difference = net_revenue - projected

    summary = {
        "charge_total": charge_total,
        "refund_total": refund_total,
        "net_revenue": net_revenue,
        "projected_revenue": projected,
        "difference": difference,
    }

    if projected and abs(difference) > tolerance * projected:
        record = {
            "type": "revenue_mismatch",
            "start_ts": start_ts,
            "end_ts": end_ts,
            **summary,
        }
        _emit_anomaly(
            record,
            write_codex,
            export_training,
            self_coding_engine,
            telemetry_feedback,
        )
        try:  # pragma: no cover - best effort
            alert_dispatcher.dispatch_alert(
                "stripe_revenue_mismatch",
                4,
                json.dumps(summary),
                summary,
            )
        except Exception:
            logger.exception("alert dispatch failed", extra=summary)
        logger.warning("stripe revenue mismatch", extra=record)
    else:
        logger.info("stripe revenue summary", extra=summary)

    return summary


def compare_revenue_window(
    start_ts: int,
    end_ts: int,
    *,
    tolerance: float = 0.1,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> Optional[Dict[str, float]]:
    """Compare Stripe revenue and ROI projections for a time window.

    This function is retained for backwards compatibility and returns details
    only when a mismatch is detected.
    """

    summary = summarize_revenue_window(
        start_ts,
        end_ts,
        tolerance=tolerance,
        write_codex=write_codex,
        export_training=export_training,
        self_coding_engine=self_coding_engine,
        telemetry_feedback=telemetry_feedback,
    )
    projected = summary.get("projected_revenue", 0.0)
    difference = summary.get("difference", 0.0)
    if projected and abs(difference) > tolerance * projected:
        return {
            "net_revenue": summary["net_revenue"],
            "projected_revenue": projected,
            "difference": difference,
        }
    return None


# Convenience wrappers used by tests and the CLI ---------------------------


def check_events(
    hours: int = 1,
    *,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> List[Dict[str, Any]]:
    """Check for missing charges within the last ``hours``."""

    api_key = load_api_key()
    if not api_key:
        return []
    end_ts = int(time.time())
    start_ts = end_ts - int(hours * 3600)
    check_webhook_endpoints(
        api_key,
        write_codex=write_codex,
        export_training=export_training,
        self_coding_engine=self_coding_engine,
        telemetry_feedback=telemetry_feedback,
    )
    # Load the entire ledger window; many tests use historical timestamps.
    ledger = load_local_ledger(0, end_ts)
    billing_logs = load_billing_logs(0, end_ts)
    allowed_accounts = _allowed_account_ids()
    expected_account_id = _expected_account_id(api_key)
    if expected_account_id:
        allowed_accounts.add(expected_account_id)
    charges = fetch_recent_charges(api_key, start_ts, end_ts)
    approved = load_approved_workflows()
    _fetch_and_apply_recent_issues()
    anomalies = detect_account_mismatches(
        charges,
        [],
        [],
        allowed_accounts,
        write_codex=write_codex,
        export_training=export_training,
        self_coding_engine=self_coding_engine,
        telemetry_feedback=telemetry_feedback,
    )
    anomalies.extend(
        detect_unauthorized_charges(
            charges,
            ledger,
            billing_logs,
            approved,
            write_codex=write_codex,
            export_training=export_training,
            self_coding_engine=self_coding_engine,
            telemetry_feedback=telemetry_feedback,
        )
    )
    anomalies.extend(
        detect_missing_charges(
            charges,
            ledger,
            billing_logs,
            approved,
            write_codex=write_codex,
            export_training=export_training,
            self_coding_engine=self_coding_engine,
            telemetry_feedback=telemetry_feedback,
        )
    )
    if anomalies and DiscrepancyDB and DiscrepancyRecord:
        try:  # pragma: no cover - best effort
            msg = f"{len(anomalies)} stripe anomalies detected"
            DiscrepancyDB().add(
                DiscrepancyRecord(message=msg, metadata={"count": len(anomalies)})
            )
        except Exception:
            logger.exception("Failed to record discrepancy summary")
    if SKIP_ANOMALY_TYPES:
        anomalies = [a for a in anomalies if a.get("type") not in SKIP_ANOMALY_TYPES]
    return anomalies


def check_revenue_projection(
    hours: int = 1,
    *,
    tolerance: float = 0.1,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> Optional[Dict[str, float]]:
    """Compare revenue for the last ``hours`` against projections."""
    end_ts = int(time.time())
    start_ts = end_ts - int(hours * 3600)
    return compare_revenue_window(
        start_ts,
        end_ts,
        tolerance=tolerance,
        write_codex=write_codex,
        export_training=export_training,
        self_coding_engine=self_coding_engine,
        telemetry_feedback=telemetry_feedback,
    )


# CLI ----------------------------------------------------------------------


# CLI can run without a context builder
def main(
    argv: Optional[List[str]] = None,
    *,
    context_builder: "ContextBuilder" | None = None,  # nocb
) -> None:
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
    parser.add_argument(
        "--export-training",
        action="store_true",
        help="write normalized anomalies to training_data/stripe_anomalies.jsonl",
    )
    args = parser.parse_args(argv)

    api_key = load_api_key()
    if not api_key:
        logger.error("Cannot run watchdog without Stripe API key")
        return
    expected_account_id = _expected_account_id(api_key)

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
        start_ts = _read_last_run_ts() or end_ts - 3600

    ANOMALY_LOG = Path(args.audit_log)
    ledger = load_local_ledger(start_ts, end_ts)
    charge_logs = load_billing_logs(start_ts, end_ts, action="charge")
    refund_logs = load_billing_logs(start_ts, end_ts, action="refund")
    failed_logs = load_billing_logs(start_ts, end_ts, action="failed")
    approved = load_approved_workflows()
    allowed_accounts = _allowed_account_ids()
    if expected_account_id:
        allowed_accounts.add(expected_account_id)

    engine = None
    telemetry = None
    builder = context_builder
    if SANITY_LAYER_FEEDBACK_ENABLED:
        if (
            SelfCodingEngine
            and CodeDB
            and MenaceMemoryManager
            and builder is not None
        ):
            try:
                builder.refresh_db_weights()
                engine = SelfCodingEngine(
                    CodeDB(), MenaceMemoryManager(), context_builder=builder
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to initialise SelfCodingEngine")
        if (
            TelemetryFeedback
            and ErrorLogger
            and engine is not None
            and builder is not None
        ):
            try:
                telemetry = TelemetryFeedback(
                    ErrorLogger(context_builder=builder), engine
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to initialise telemetry feedback")

    charges = fetch_recent_charges(api_key, start_ts, end_ts)
    refunds = fetch_recent_refunds(api_key, start_ts, end_ts)
    events = fetch_recent_events(api_key, start_ts, end_ts)
    _fetch_and_apply_recent_issues()
    detect_account_mismatches(
        charges,
        refunds,
        events,
        allowed_accounts,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    detect_unauthorized_charges(
        charges,
        ledger,
        charge_logs,
        approved,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    detect_missing_charges(
        charges,
        ledger,
        charge_logs,
        approved,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    detect_unauthorized_refunds(
        refunds,
        ledger,
        refund_logs,
        approved,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    detect_missing_refunds(
        refunds,
        ledger,
        refund_logs,
        approved,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    detect_unauthorized_failures(
        events,
        ledger,
        failed_logs,
        approved,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    detect_failed_events(
        events,
        ledger,
        failed_logs,
        approved,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    check_webhook_endpoints(
        api_key,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    compare_revenue_window(
        start_ts,
        end_ts,
        write_codex=args.write_codex,
        export_training=args.export_training,
        self_coding_engine=engine,
        telemetry_feedback=telemetry,
    )
    _write_last_run_ts(end_ts)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    logging.basicConfig(level=logging.INFO)
    main()
