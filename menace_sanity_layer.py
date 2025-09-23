from __future__ import annotations

"""Utility for recording payment and billing anomalies."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dynamic_path_router import resolve_path

import audit_logger
from log_tags import FEEDBACK, ERROR_FIX
import db_router
from db_router import LOCAL_TABLES

try:  # Optional dependency – required for payment safety checks
    import stripe_billing_router  # noqa: F401
except Exception:  # pragma: no cover - best effort
    stripe_billing_router = None  # type: ignore

try:  # pragma: no cover - prefer package import
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback when not imported as package
    import importlib.util
    import sys
    from types import ModuleType

    module_path = resolve_path("unified_event_bus.py")
    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.unified_event_bus", module_path
    )
    module = importlib.util.module_from_spec(spec)
    pkg = ModuleType("menace_sandbox")
    pkg.__path__ = [str(module_path.parent)]  # type: ignore[attr-defined]
    sys.modules.setdefault("menace_sandbox", pkg)
    sys.modules["menace_sandbox.unified_event_bus"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    UnifiedEventBus = module.UnifiedEventBus  # type: ignore

try:  # Optional dependency – discrepancy database
    from failure_learning_system import DiscrepancyDB
except Exception:  # pragma: no cover - best effort
    DiscrepancyDB = None  # type: ignore

# Dedicated discrepancy DB for billing events
try:  # pragma: no cover - optional
    from discrepancy_db import (
        DiscrepancyDB as BillingDiscrepancyDB,
        DiscrepancyRecord,
    )
except Exception:  # pragma: no cover - best effort
    BillingDiscrepancyDB = None  # type: ignore
    DiscrepancyRecord = None  # type: ignore

try:  # Optional dependency – GPT memory manager
    from menace_sandbox.gpt_memory import GPTMemoryManager
except Exception:  # pragma: no cover - best effort
    GPTMemoryManager = None  # type: ignore

try:  # Optional dependency – shared GPT memory
    from shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - best effort
    GPT_MEMORY_MANAGER = None  # type: ignore

MenaceMemoryManager = None  # type: ignore

logger = logging.getLogger(__name__)

# Reuse a single DiscrepancyDB instance when available
_DISCREPANCY_DB = DiscrepancyDB() if DiscrepancyDB is not None else None

# Billing events leverage a dedicated DiscrepancyDB when available
_BILLING_EVENT_DB = (
    BillingDiscrepancyDB() if BillingDiscrepancyDB is not None else None
)

_GPT_MEMORY: GPTMemoryManager | None = None

# Mapping of event types to corrective instructions.
EVENT_TYPE_INSTRUCTIONS: Dict[str, str] = {
    "missing_charge": (
        "Avoid creating Stripe charges without billing log entries or central routing."
    ),
    "missing_refund": (
        "Avoid issuing refunds without corresponding ledger entries."
    ),
    "missing_failure_log": (
        "Avoid handling Stripe failures without logging the event."
    ),
    "unapproved_workflow": (
        "Avoid running Stripe workflows without explicit approval."
    ),
    "unknown_webhook": (
        "Avoid unregistered Stripe webhooks; register endpoints before use."
    ),
    "disabled_webhook": (
        "Avoid relying on disabled Stripe webhook endpoints."
    ),
    "revenue_mismatch": (
        "Avoid revenue figures that diverge from ledger or ROI projections."
    ),
    "account_mismatch": (
        "Avoid routing Stripe activity to unapproved accounts."
    ),
    "unauthorized_charge": (
        "Avoid processing Stripe charges without explicit authorization."
    ),
    "unauthorized_refund": (
        "Avoid issuing Stripe refunds without explicit authorization."
    ),
    "unauthorized_failure": (
        "Avoid processing Stripe payment failures without explicit authorization."
    ),
}

# Optional overrides loaded from ``config/billing_instructions.yaml``.  The file
# is read once and cached until :func:`refresh_billing_instructions` is invoked
# by modules monitoring configuration changes.
_INSTRUCTION_PATH = Path(resolve_path("config/billing_instructions.yaml"))
_INSTRUCTION_OVERRIDES: Dict[str, str] | None = None

# Default threshold and hint settings; values can be overridden via the
# configuration file.  ``PAYMENT_ANOMALY_THRESHOLD`` acts as the global fallback
# while ``ANOMALY_THRESHOLDS`` allows per-event overrides.
_DEFAULT_PAYMENT_ANOMALY_THRESHOLD = 3

DEFAULT_ANOMALY_HINTS: Dict[str, Dict[str, Any]] = {
    "missing_charge": {"block_unlogged_charges": True},
    "missing_refund": {"block_unlogged_refunds": True},
    "missing_failure_log": {"log_stripe_failures": True},
    "unapproved_workflow": {"enforce_workflow_approval": True},
    "unknown_webhook": {"register_stripe_webhooks": True},
    "disabled_webhook": {"reactivate_stripe_webhook": True},
    "revenue_mismatch": {"reconcile_revenue": True},
    "account_mismatch": {"verify_stripe_account": True},
    "unauthorized_charge": {"block_unauthorized_charges": True},
    "unauthorized_refund": {"block_unauthorized_refunds": True},
    "unauthorized_failure": {"block_unauthorized_failures": True},
}

PAYMENT_ANOMALY_THRESHOLD = _DEFAULT_PAYMENT_ANOMALY_THRESHOLD
ANOMALY_HINTS: Dict[str, Dict[str, Any]] = DEFAULT_ANOMALY_HINTS.copy()
ANOMALY_THRESHOLDS: Dict[str, int] = {}

# ---------------------------------------------------------------------------
# Billing anomaly utilities

# Ensure the new table is recognised by the DB router
LOCAL_TABLES.add("billing_anomalies")

_EVENT_BUS = UnifiedEventBus()


_MEMORY_MANAGER: MenaceMemoryManager | None = None

# ---------------------------------------------------------------------------
# Anomaly suppression configuration
# ---------------------------------------------------------------------------

_SUPPRESSION_CONFIG_PATH = Path(resolve_path("config/stripe_watchdog.yaml"))

# Default suppression settings – events below ``severity_threshold`` are ignored
# and identical events are only forwarded ``max_occurrences`` times within
# ``window_seconds``.
_SUPPRESSION_SETTINGS: Dict[str, float] = {
    "window_seconds": 60.0,
    "max_occurrences": 1.0,
    "severity_threshold": 0.0,
}


def _load_suppression_settings(path: Path | None = None) -> None:
    """Load anomaly suppression settings from YAML config."""

    cfg = path or _SUPPRESSION_CONFIG_PATH
    try:
        with cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        sup = data.get("anomaly_suppression", {})
        _SUPPRESSION_SETTINGS["window_seconds"] = float(
            sup.get("window_seconds", _SUPPRESSION_SETTINGS["window_seconds"])
        )
        _SUPPRESSION_SETTINGS["max_occurrences"] = float(
            sup.get("max_occurrences", _SUPPRESSION_SETTINGS["max_occurrences"])
        )
        _SUPPRESSION_SETTINGS["severity_threshold"] = float(
            sup.get(
                "severity_threshold", _SUPPRESSION_SETTINGS["severity_threshold"]
            )
        )
    except FileNotFoundError:
        pass
    except Exception:  # pragma: no cover - best effort
        logger.exception(
            "failed to load anomaly suppression config", extra={"path": str(cfg)}
        )


_load_suppression_settings()

# Cache of recent anomalies keyed by (event_type, metadata_json)
_ANOMALY_CACHE: Dict[tuple[str, str], List[float]] = {}


def _allow_event(event_type: str, metadata: Dict[str, Any], severity: float) -> bool:
    """Return ``True`` if the event passes severity and dedup filters."""

    if severity < _SUPPRESSION_SETTINGS["severity_threshold"]:
        return False
    now = time.time()
    key = (event_type, json.dumps(metadata, sort_keys=True))
    window = _SUPPRESSION_SETTINGS["window_seconds"]
    max_occ = int(_SUPPRESSION_SETTINGS["max_occurrences"])
    times = [t for t in _ANOMALY_CACHE.get(key, []) if now - t < window]
    if len(times) >= max_occ:
        _ANOMALY_CACHE[key] = times
        return False
    times.append(now)
    _ANOMALY_CACHE[key] = times
    return True


def _get_memory_manager() -> MenaceMemoryManager | None:
    """Lazily construct the shared :class:`MenaceMemoryManager`."""
    global _MEMORY_MANAGER, MenaceMemoryManager
    if _MEMORY_MANAGER is None:
        if MenaceMemoryManager is None:
            try:  # pragma: no cover - import lazily to avoid heavy deps at import time
                from menace_memory_manager import MenaceMemoryManager as _MMM

                MenaceMemoryManager = _MMM
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to import MenaceMemoryManager")
                return None
        try:
            _MEMORY_MANAGER = MenaceMemoryManager()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to initialise MenaceMemoryManager")
            _MEMORY_MANAGER = None
    return _MEMORY_MANAGER


def _get_gpt_memory() -> GPTMemoryManager | None:
    """Lazily construct :class:`GPTMemoryManager` for feedback logging."""

    global _GPT_MEMORY
    if _GPT_MEMORY is None and GPTMemoryManager is not None:
        try:  # pragma: no cover - avoid heavy init in tests
            _GPT_MEMORY = GPTMemoryManager()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to initialise GPTMemoryManager")
            _GPT_MEMORY = None
    return _GPT_MEMORY


def _load_instruction_overrides() -> Dict[str, str]:
    """Return mapping of event type overrides loaded from config."""

    global _INSTRUCTION_OVERRIDES, PAYMENT_ANOMALY_THRESHOLD
    if _INSTRUCTION_OVERRIDES is None:
        data: Dict[str, Any]
        try:
            with _INSTRUCTION_PATH.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            data = {}
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "failed to load billing instructions", extra={"path": str(_INSTRUCTION_PATH)}
            )
            data = {}

        instructions = data.get("instructions")
        if instructions is None:
            instructions = {k: v for k, v in data.items() if isinstance(v, str)}
        _INSTRUCTION_OVERRIDES = {str(k): str(v) for k, v in (instructions or {}).items()}

        try:
            PAYMENT_ANOMALY_THRESHOLD = int(
                data.get("payment_anomaly_threshold", PAYMENT_ANOMALY_THRESHOLD)
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("invalid payment anomaly threshold", exc_info=True)

        ANOMALY_HINTS.update(
            {
                str(k): dict(v)
                for k, v in (data.get("anomaly_hints", {}) or {}).items()
                if isinstance(v, dict)
            }
        )

        ANOMALY_THRESHOLDS.clear()
        for k, v in (data.get("anomaly_thresholds", {}) or {}).items():
            try:
                ANOMALY_THRESHOLDS[str(k)] = int(v)
            except Exception:  # pragma: no cover - best effort
                logger.exception("invalid anomaly threshold", extra={"event_type": k})

    return _INSTRUCTION_OVERRIDES


def refresh_billing_instructions(path: Path | str | None = None) -> None:
    """Clear cached instruction overrides so they're reloaded on next use."""

    global _INSTRUCTION_OVERRIDES, _INSTRUCTION_PATH, PAYMENT_ANOMALY_THRESHOLD
    global ANOMALY_HINTS, ANOMALY_THRESHOLDS
    if path is not None:
        _INSTRUCTION_PATH = Path(path)
    _INSTRUCTION_OVERRIDES = None
    PAYMENT_ANOMALY_THRESHOLD = _DEFAULT_PAYMENT_ANOMALY_THRESHOLD
    ANOMALY_HINTS = DEFAULT_ANOMALY_HINTS.copy()
    ANOMALY_THRESHOLDS = {}


def _anomaly_instruction(
    event_type: str, metadata: Dict[str, Any], instruction: str | None
) -> str:
    """Generate a concise instruction string for an anomaly."""

    overrides = _load_instruction_overrides()
    instr = (
        overrides.get(event_type)
        or instruction
        or metadata.get("instruction")
        or metadata.get("description")
        or ""
    )
    if not instr:
        details = ", ".join(f"{k}={v}" for k, v in sorted(metadata.items()))
        instr = f"Avoid {event_type} anomalies {details}".strip()
    instr = instr.strip()
    if not instr.lower().startswith("avoid"):
        instr = "Avoid " + instr
    if not instr.endswith("."):
        instr += "."
    return instr


def record_event(
    event_type: str,
    metadata: Dict[str, Any],
    *,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> None:
    """Record a generic anomaly event and log corrective guidance.

    The ``event_type`` is mapped to a human-readable instruction which, along with
    the provided ``metadata``, is persisted to :class:`gpt_memory.GPTMemoryManager`
    using the :data:`~log_tags.FEEDBACK` and :data:`~log_tags.ERROR_FIX` tags.  When
    supplied, ``self_coding_engine`` and ``telemetry_feedback`` hooks allow repeated
    anomalies to influence future code-generation or architectural decisions.
    """

    instruction = _anomaly_instruction(
        event_type, metadata, EVENT_TYPE_INSTRUCTIONS.get(event_type)
    )

    mgr = _get_gpt_memory()
    if mgr is not None:
        try:  # pragma: no cover - best effort
            mgr.log_interaction(
                instruction,
                json.dumps(
                    {"event_type": event_type, "metadata": metadata},
                    sort_keys=True,
                ),
                tags=[FEEDBACK, ERROR_FIX, event_type],
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "GPT memory logging failed", extra={"event_type": event_type}
            )

    if self_coding_engine is not None:
        try:  # pragma: no cover - best effort
            update_fn = getattr(
                self_coding_engine, "update_generation_params", None
            )
            if callable(update_fn):
                changes = update_fn(metadata) or {}
                if changes:
                    logger.info(
                        "generation params updated", extra={"changes": changes}
                    )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "self_coding_engine hook failed", extra={"event_type": event_type}
            )

    if telemetry_feedback is not None:
        try:  # pragma: no cover - best effort
            feedback_fn = getattr(
                telemetry_feedback, "record_event", None
            ) or getattr(telemetry_feedback, "log_event", None)
            if callable(feedback_fn):
                feedback_fn(event_type, metadata)
            check_fn = getattr(telemetry_feedback, "check", None)
            if callable(check_fn):
                check_fn()
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "telemetry_feedback hook failed",
                extra={"event_type": event_type},
            )


def publish_anomaly(event: dict, *, _check: bool = True) -> bool:
    """Publish *event* to the ``billing.anomaly`` topic.

    Returns ``True`` when the event is forwarded.  When ``_check`` is ``True``
    (the default) the suppression cache and severity threshold are consulted
    before emitting the event.  :func:`record_billing_anomaly` performs its own
    filtering and therefore calls this function with ``_check=False``.
    """

    if _check and not _allow_event(
        event.get("event_type", ""), event.get("metadata", {}), float(event.get("severity", 0.0))
    ):
        return False
    try:
        _EVENT_BUS.publish("billing.anomaly", event)
        return True
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to publish billing anomaly")
        return False


def record_billing_anomaly(
    event_type: str,
    metadata: Dict[str, Any],
    *,
    severity: float = 1.0,
    source_workflow: str | None = None,
    publish: bool = True,
) -> None:
    """Persist a billing anomaly to SQLite and optionally publish it."""

    severity = float(severity)
    if not _allow_event(event_type, metadata, severity):
        return
    if db_router.GLOBAL_ROUTER is None:
        raise RuntimeError("Database router is not initialised")
    ts = time.time()
    db_router.GLOBAL_ROUTER.execute_and_log(
        "billing_anomalies",
        (
            "CREATE TABLE IF NOT EXISTS billing_anomalies("
            "event_type TEXT, metadata TEXT, ts REAL, "
            "severity REAL, source_workflow TEXT)"
        ),
    )
    db_router.GLOBAL_ROUTER.execute_and_log(
        "billing_anomalies",
        (
            "INSERT INTO billing_anomalies("
            "event_type, metadata, ts, severity, source_workflow) "
            "VALUES (?,?,?,?,?)"
        ),
        (
            event_type,
            json.dumps(metadata, sort_keys=True),
            ts,
            severity,
            source_workflow,
        ),
    )

    event = {
        "event_type": event_type,
        "metadata": metadata,
        "ts": ts,
        "severity": severity,
        "source_workflow": source_workflow,
    }
    if publish:
        publish_anomaly(event, _check=False)


def list_anomalies(limit: int = 20) -> List[dict]:
    """Return the most recent stored billing anomalies."""

    if db_router.GLOBAL_ROUTER is None:
        raise RuntimeError("Database router is not initialised")
    db_router.GLOBAL_ROUTER.execute_and_log(
        "billing_anomalies",
        (
            "CREATE TABLE IF NOT EXISTS billing_anomalies("
            "event_type TEXT, metadata TEXT, ts REAL, "
            "severity REAL, source_workflow TEXT)"
        ),
    )
    rows = db_router.GLOBAL_ROUTER.execute_and_log(
        "billing_anomalies",
        (
            "SELECT event_type, metadata, ts, severity, source_workflow "
            "FROM billing_anomalies ORDER BY ts DESC LIMIT ?"
        ),
        (limit,),
    )
    result: List[dict] = []
    for event_type, meta_json, ts, severity, source in rows:
        try:
            meta = json.loads(meta_json)
        except Exception:  # pragma: no cover - best effort
            meta = {"_corrupt": meta_json}
        result.append(
            {
                "event_type": event_type,
                "metadata": meta,
                "ts": ts,
                "severity": severity,
                "source_workflow": source,
            }
        )
    return result


def record_payment_anomaly(
    event_type: str,
    metadata: Dict[str, Any],
    instruction: str | None = None,
    *,
    severity: float = 1.0,
    write_codex: bool = False,
    export_training: bool = False,
    self_coding_engine: Any | None = None,
    telemetry_feedback: Any | None = None,
) -> None:
    """Persist anomaly details, memory feedback and audit trail.

    Parameters
    ----------
    event_type:
        Classification of the anomaly.
    metadata:
        Additional details describing the event.
    instruction:
        Optional guidance associated with the anomaly.  If omitted a
        concise instruction is generated automatically.
    severity:
        Importance level of the anomaly; defaults to ``1.0``.
    write_codex:
        Whether the anomaly should be emitted as a Codex training sample.
    export_training:
        Whether the anomaly should be exported for training datasets.
    self_coding_engine:
        Optional hook allowing repeated anomalies to influence generation
        parameters.
    telemetry_feedback:
        Optional hook for emitting telemetry events and performing health
        checks related to the anomaly.
    """

    instruction = _anomaly_instruction(event_type, metadata, instruction)
    meta = {
        **metadata,
        "write_codex": write_codex,
        "export_training": export_training,
    }
    if _DISCREPANCY_DB is not None:
        try:
            _DISCREPANCY_DB.log_detection(
                event_type, severity, json.dumps(meta, sort_keys=True)
            )
        except Exception:
            logger.exception(
                "failed to log detection", extra={"event_type": event_type, "metadata": metadata}
            )

    if GPT_MEMORY_MANAGER is not None:
        try:
            GPT_MEMORY_MANAGER.log_interaction(
                instruction,
                json.dumps(
                    {
                        "event_type": event_type,
                        "metadata": meta,
                        "severity": severity,
                    },
                    sort_keys=True,
                ),
                tags=[FEEDBACK, instruction],
            )
        except Exception:
            logger.exception("memory logging failed", extra={"instruction": instruction})

    mm = _get_memory_manager()
    trigger_correction = False
    if mm is not None:
        try:
            key = f"billing:{event_type}"
            data = {"instruction": instruction, "count": 1}
            existing = mm.query(key, 1)
            prev_count = 0
            if existing:
                try:
                    existing_data = json.loads(existing[0].data)
                    prev_count = int(existing_data.get("count", 0))
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to parse existing memory entry", extra={"key": key}
                    )
            data["count"] = prev_count + 1
            mm.store(key, data, tags="billing,anomaly")
            threshold = ANOMALY_THRESHOLDS.get(event_type, PAYMENT_ANOMALY_THRESHOLD)
            if data["count"] >= threshold and prev_count < threshold:
                trigger_correction = True
        except Exception:  # pragma: no cover - best effort
            logger.exception("MenaceMemoryManager logging failed")

    if trigger_correction and self_coding_engine is not None:
        try:  # pragma: no cover - best effort
            update_fn = getattr(self_coding_engine, "update_generation_params", None)
            if callable(update_fn):
                hint = ANOMALY_HINTS.get(event_type, {"block_unlogged_charges": True})
                hint = {**hint, "event_type": event_type}
                update_fn(hint)
        except Exception:  # pragma: no cover - best effort
            logger.exception("self_coding_engine update failed")

    if telemetry_feedback is not None:
        try:  # pragma: no cover - best effort
            feedback_fn = getattr(
                telemetry_feedback, "record_event", None
            ) or getattr(telemetry_feedback, "log_event", None)
            if callable(feedback_fn):
                feedback_fn(event_type, meta)
            check_fn = getattr(telemetry_feedback, "check", None)
            if callable(check_fn):
                check_fn()
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "telemetry_feedback hook failed", extra={"event_type": event_type}
            )

    try:
        audit_logger.log_event(
            "payment_sanity",
            {
                "event_type": event_type,
                "metadata": meta,
                "instruction": instruction,
                "severity": severity,
                "write_codex": write_codex,
                "export_training": export_training,
            },
        )
    except Exception:
        logger.exception("audit logging failed")


def record_billing_event(
    event_type: str,
    metadata: Dict[str, Any],
    instruction: str,
    *,
    config_path: str | Path | None = None,
    self_coding_engine: Any | None = None,
) -> None:
    """Persist a billing event and capture corrective guidance.

    The event is stored in :class:`discrepancy_db.DiscrepancyDB` when available
    and the ``instruction`` is logged to :class:`gpt_memory.GPTMemoryManager`
    with the :data:`~log_tags.FEEDBACK` tag.  Optional ``config_path`` and
    ``self_coding_engine`` hooks allow callers to adjust code-generation
    parameters based on the event details.
    """

    if _BILLING_EVENT_DB is not None and DiscrepancyRecord is not None:
        try:
            rec = DiscrepancyRecord(
                message=event_type,
                metadata={"instruction": instruction, **metadata},
            )
            _BILLING_EVENT_DB.add(rec)
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "failed to persist billing event",
                extra={"event_type": event_type, "metadata": metadata},
            )

    mgr = _get_gpt_memory()
    if mgr is not None:
        try:
            mgr.log_interaction(
                instruction,
                json.dumps(
                    {"event_type": event_type, "metadata": metadata},
                    sort_keys=True,
                ),
                tags=[FEEDBACK, "billing"],
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "GPT memory logging failed", extra={"instruction": instruction}
            )

    if config_path:
        try:
            path = Path(config_path)
            existing: Dict[str, Any] = {}
            if path.exists():
                existing = json.loads(path.read_text() or "{}")
            updates = metadata.get("config_updates", {})
            if isinstance(updates, dict):
                existing.update(updates)
            path.write_text(json.dumps(existing, indent=2, sort_keys=True))
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "failed to update config file", extra={"path": str(config_path)}
            )

    trigger_correction = False
    mm = _get_memory_manager()
    if mm is not None:
        try:
            key = f"billing:event:{event_type}"
            data = {"instruction": instruction, "count": 1}
            existing = mm.query(key, 1)
            prev_count = 0
            if existing:
                try:
                    existing_data = json.loads(existing[0].data)
                    prev_count = int(existing_data.get("count", 0))
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to parse existing memory entry", extra={"key": key}
                    )
            data["count"] = prev_count + 1
            mm.store(key, data, tags="billing,event")
            threshold = ANOMALY_THRESHOLDS.get(event_type, PAYMENT_ANOMALY_THRESHOLD)
            if data["count"] >= threshold and prev_count < threshold:
                trigger_correction = True
        except Exception:  # pragma: no cover - best effort
            logger.exception("MenaceMemoryManager logging failed")

    if trigger_correction and self_coding_engine is not None:
        try:
            update_fn = getattr(self_coding_engine, "update_generation_params", None)
            if callable(update_fn):
                hint = ANOMALY_HINTS.get(event_type, {})
                if hint:
                    update_fn({**hint, "event_type": event_type})
        except Exception:  # pragma: no cover - best effort
            logger.exception("self_coding_engine update failed")


def fetch_recent_billing_issues(limit: int = 5) -> List[str]:
    """Return recent billing-related feedback snippets."""

    mgr = _get_gpt_memory()
    if mgr is None:
        return []
    try:
        records = mgr.retrieve("billing", limit=limit, tags=[FEEDBACK, "billing"])
        return [r.prompt for r in records if getattr(r, "prompt", None)]
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to fetch billing issues")
        return []


__all__ = [
    "record_event",
    "record_payment_anomaly",
    "record_billing_anomaly",
    "record_billing_event",
    "fetch_recent_billing_issues",
    "publish_anomaly",
    "list_anomalies",
]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect billing anomalies")
    sub = parser.add_subparsers(dest="cmd")
    list_p = sub.add_parser("list", help="List stored anomalies")
    list_p.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.cmd == "list":
        for row in list_anomalies(args.limit):
            print(json.dumps(row, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI tool
    _main()
