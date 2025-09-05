from __future__ import annotations

"""Utility for recording payment anomalies and feedback."""

import json
import logging
from typing import Any, Dict

import audit_logger
from failure_learning_system import DiscrepancyDB
from log_tags import FEEDBACK
from shared_gpt_memory import GPT_MEMORY_MANAGER

logger = logging.getLogger(__name__)

# Reuse a single DiscrepancyDB instance
_DISCREPANCY_DB = DiscrepancyDB()


def record_payment_anomaly(
    event_type: str,
    metadata: Dict[str, Any],
    instruction: str,
    *,
    severity: float = 1.0,
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
    """

    try:
        _DISCREPANCY_DB.log_detection(
            event_type, severity, json.dumps(metadata, sort_keys=True)
        )
    except Exception:
        logger.exception(
            "failed to log detection", extra={"event_type": event_type, "metadata": metadata}
        )

    try:
        GPT_MEMORY_MANAGER.log_interaction(
            instruction,
            json.dumps(
                {
                    "event_type": event_type,
                    "metadata": metadata,
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
                "metadata": metadata,
                "instruction": instruction,
                "severity": severity,
            },
        )
    except Exception:
        logger.exception("audit logging failed")


__all__ = ["record_payment_anomaly"]
