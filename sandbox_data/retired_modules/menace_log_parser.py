from __future__ import annotations

"""Secure parser for Menace action logs."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

LOG_PATH = "/mnt/shared/menace_logs/actions.jsonl"
ERROR_LOG = "parser_errors.log"


REQUIRED_FIELDS = [
    "timestamp",
    "action_type",
    "target_domain",
    "risk_score",
    "alignment_score",
    "source_file",
    "additional_metadata",
]


def validate_log_entry(entry: Dict[str, Any]) -> bool:
    """Return ``True`` if *entry* contains valid fields."""

    for key in REQUIRED_FIELDS:
        if key not in entry:
            return False

    if not isinstance(entry["timestamp"], (int, float)):
        try:
            float(entry["timestamp"])
        except Exception:
            return False
    if not isinstance(entry["action_type"], str):
        return False
    if not isinstance(entry["target_domain"], str):
        return False
    if not isinstance(entry["risk_score"], (int, float)):
        return False
    if not isinstance(entry["alignment_score"], (int, float)):
        return False
    if not isinstance(entry["source_file"], str):
        return False
    if not isinstance(entry["additional_metadata"], dict):
        return False

    return True


def load_unprocessed_logs(last_timestamp: float) -> List[Dict[str, Any]]:
    """Return logs newer than *last_timestamp* sorted by ``timestamp``."""

    entries: List[Dict[str, Any]] = []
    errors: List[str] = []

    if not os.path.exists(LOG_PATH):
        return entries

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"Malformed JSON: {exc} - {line.strip()}")
                    continue
                if not validate_log_entry(entry):
                    errors.append(f"Invalid entry: {line.strip()}")
                    continue
                try:
                    ts = float(entry["timestamp"])
                except Exception:
                    errors.append(f"Bad timestamp: {line.strip()}")
                    continue
                if ts > last_timestamp:
                    entries.append(entry)
    except Exception as exc:
        errors.append(f"Error reading log file: {exc}")

    if errors:
        try:
            with open(ERROR_LOG, "a", encoding="utf-8") as err:
                for msg in errors:
                    err.write(msg + "\n")
        except Exception:
            pass

    entries.sort(key=lambda e: float(e["timestamp"]))
    return entries


def extract_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a condensed representation of *entry*."""

    return {
        "timestamp": entry["timestamp"],
        "domain": entry["target_domain"],
        "risk_score": entry["risk_score"],
        "action_type": entry["action_type"],
    }


__all__ = [
    "load_unprocessed_logs",
    "validate_log_entry",
    "extract_summary",
]
