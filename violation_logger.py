"""Append-only violation logging utilities for Security AI."""

from __future__ import annotations

import json
import os
import time
import logging
from datetime import datetime
from threading import Thread
from typing import Any, List, Dict, Optional

from .retry_utils import publish_with_retry

try:  # optional dependency
    from .unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - bus optional
    UnifiedEventBus = None  # type: ignore

# Directory and file path for violation logs
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_PATH = os.path.join(LOG_DIR, "violation_log.jsonl")

logger = logging.getLogger(__name__)

_event_bus: Optional[UnifiedEventBus]
if UnifiedEventBus is not None:
    try:
        _event_bus = UnifiedEventBus()
    except Exception:  # pragma: no cover - runtime optional
        _event_bus = None
else:  # pragma: no cover - bus not available
    _event_bus = None


def set_event_bus(bus: Optional[UnifiedEventBus]) -> None:
    """Override the global event bus instance."""
    global _event_bus
    _event_bus = bus


def _ensure_log_dir() -> None:
    """Create the log directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_violation(
    entry_id: str,
    violation_type: str,
    severity: int,
    evidence: Dict[str, Any],
    *,
    alignment_warning: bool = False,
) -> None:
    """Record a violation or alignment warning in the append-only log.

    Parameters
    ----------
    entry_id: str
        Unique identifier of the action or event that triggered the violation.
    violation_type: str
        Category of violation, e.g. ``"policy_violation"`` or ``"security_risk"``.
    severity: int
        Integer severity score. Higher values indicate more serious violations.
    evidence: dict
        Contextual information showing why the event was flagged. Should be
        JSON serialisable.
    alignment_warning: bool, optional
        When ``True`` the entry represents an alignment warning rather than a
        hard violation.  Warnings are also published on the event bus.
    """
    _ensure_log_dir()
    record = {
        "timestamp": int(time.time()),
        "entry_id": entry_id,
        "violation_type": violation_type,
        "severity": severity,
        "evidence": evidence,
        "alignment_warning": alignment_warning,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    if alignment_warning and _event_bus is not None:
        payload = {
            "timestamp": record["timestamp"],
            "entry_id": entry_id,
            "violation_type": violation_type,
            "severity": severity,
            "evidence": evidence,
        }

        def _publish() -> None:
            try:
                publish_with_retry(
                    _event_bus, "alignment:flag", payload, delay=0.1
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("failed publishing alignment warning: %s", exc)

        Thread(target=_publish, daemon=True).start()


def load_recent_violations(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent *limit* violation records."""
    if limit <= 0:
        return []
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    records: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_recent_alignment_warnings(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent *limit* alignment warnings."""
    if limit <= 0:
        return []
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    warnings: List[Dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("alignment_warning"):
            warnings.append(rec)
            if len(warnings) >= limit:
                break
    warnings.reverse()
    return warnings


def recent_alignment_warnings(limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience wrapper returning recent alignment warnings.

    This helper provides a shorter public name while delegating to
    :func:`load_recent_alignment_warnings` for the actual retrieval logic.
    """

    return load_recent_alignment_warnings(limit)


def violation_summary(entry_id: str) -> str:
    """Return a brief summary of all violations for *entry_id*."""
    if not os.path.exists(LOG_PATH):
        return "No violations logged."
    summaries = []
    with open(LOG_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("entry_id") == entry_id:
                ts = datetime.utcfromtimestamp(rec.get("timestamp", 0)).isoformat()
                vt = rec.get("violation_type")
                sev = rec.get("severity")
                summaries.append(f"{ts} - {vt} (severity {sev})")
    if not summaries:
        return f"No violations found for {entry_id}."
    joined = "\n".join(summaries)
    return f"Violation report for {entry_id}:\n{joined}"


__all__ = [
    "log_violation",
    "load_recent_violations",
    "load_recent_alignment_warnings",
    "recent_alignment_warnings",
    "violation_summary",
    "set_event_bus",
]

