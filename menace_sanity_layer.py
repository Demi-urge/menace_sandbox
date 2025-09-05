from __future__ import annotations

"""Utility for recording payment and billing anomalies."""

import json
import logging
import time
from typing import Any, Dict, List

import audit_logger
from log_tags import FEEDBACK
import db_router
from db_router import LOCAL_TABLES

try:  # pragma: no cover - prefer package import
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback when not imported as package
    import importlib.util
    import sys
    from types import ModuleType
    from pathlib import Path

    module_path = Path(__file__).with_name("unified_event_bus.py")
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

try:  # Optional dependency – shared GPT memory
    from shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - best effort
    GPT_MEMORY_MANAGER = None  # type: ignore

logger = logging.getLogger(__name__)

# Reuse a single DiscrepancyDB instance when available
_DISCREPANCY_DB = DiscrepancyDB() if DiscrepancyDB is not None else None

# ---------------------------------------------------------------------------
# Billing anomaly utilities

# Ensure the new table is recognised by the DB router
LOCAL_TABLES.add("billing_anomalies")

_EVENT_BUS = UnifiedEventBus()


def publish_anomaly(event: dict) -> None:
    """Publish *event* to the ``billing.anomaly`` topic."""
    try:
        _EVENT_BUS.publish("billing.anomaly", event)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to publish billing anomaly")


def record_billing_anomaly(
    event_type: str,
    metadata: Dict[str, Any],
    *,
    severity: float = 1.0,
    source_workflow: str | None = None,
    publish: bool = True,
) -> None:
    """Persist a billing anomaly to SQLite and optionally publish it."""

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
        publish_anomaly(event)


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
    instruction: str,
    *,
    severity: float = 1.0,
    write_codex: bool = False,
    export_training: bool = False,
) -> None:
    """Persist anomaly details, memory feedback and audit trail.

    Parameters
    ----------
    event_type:
        Classification of the anomaly.
    metadata:
        Additional details describing the event.
    instruction:
        Guidance or remediation note associated with the anomaly.
    severity:
        Importance level of the anomaly; defaults to ``1.0``.
    write_codex:
        Whether the anomaly should be emitted as a Codex training sample.
    export_training:
        Whether the anomaly should be exported for training datasets.
    """

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


__all__ = [
    "record_payment_anomaly",
    "record_billing_anomaly",
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
