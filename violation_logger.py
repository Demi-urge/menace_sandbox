"""Append-only violation logging utilities for Security AI."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict

# Directory and file path for violation logs
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_PATH = os.path.join(LOG_DIR, "violation_log.jsonl")


def _ensure_log_dir() -> None:
    """Create the log directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_violation(entry_id: str, violation_type: str, severity: int, evidence: Dict[str, Any]) -> None:
    """Record a violation event in the append-only log.

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
    """
    _ensure_log_dir()
    record = {
        "timestamp": int(time.time()),
        "entry_id": entry_id,
        "violation_type": violation_type,
        "severity": severity,
        "evidence": evidence,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


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


__all__ = ["log_violation", "load_recent_violations", "violation_summary"]

