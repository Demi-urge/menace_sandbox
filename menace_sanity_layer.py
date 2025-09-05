from __future__ import annotations

"""Utility for recording payment anomalies and feedback."""

import json
import logging
from typing import Any, Dict

import audit_logger
from log_tags import FEEDBACK

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


__all__ = ["record_payment_anomaly"]
