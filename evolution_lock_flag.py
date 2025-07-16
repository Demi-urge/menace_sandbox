from __future__ import annotations

"""Evolution lock mechanism for Security AI.

This module exposes simple file-based locking to halt Menace evolution cycles.
Security AI or other external oversight systems can trigger a lock by writing a
flag file that Menace checks before any self-evolution step. The lock remains
active until manually cleared via :func:`clear_lock`.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Paths
FLAG_DIR = "/mnt/shared/security_ai"
FLAG_PATH = os.path.join(FLAG_DIR, "lock_evolution.flag")
HISTORY_LOG = os.path.join("logs", "lock_history.jsonl")


def _ensure_dirs() -> None:
    """Create required directories."""
    os.makedirs(FLAG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_LOG), exist_ok=True)


# ---------------------------------------------------------------------------
# Public API


def trigger_lock(reason: str, severity: int) -> Dict[str, Any]:
    """Write a lock file instructing Menace to halt evolution.

    Parameters
    ----------
    reason:
        Human readable explanation for the lock.
    severity:
        Integer severity score associated with the event.

    Returns
    -------
    dict
        The record written to the flag file.
    """
    _ensure_dirs()
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reason": reason,
        "severity": int(severity),
        "triggered_by": "SecurityAI",
    }
    try:
        with open(FLAG_PATH, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
    except Exception:
        # In a failure scenario, attempt best-effort logging
        pass
    try:
        with open(HISTORY_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass
    return record


def is_lock_active() -> bool:
    """Return ``True`` if the evolution lock is currently active."""
    return os.path.exists(FLAG_PATH) and os.path.getsize(FLAG_PATH) > 0


def read_lock_data() -> Dict[str, Any]:
    """Return parsed contents of the lock file if present."""
    if not os.path.exists(FLAG_PATH):
        return {}
    try:
        with open(FLAG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def clear_lock() -> bool:
    """Remove the evolution lock file.

    Returns ``True`` if the file was removed or did not exist.
    """
    if os.path.exists(FLAG_PATH):
        try:
            os.remove(FLAG_PATH)
            return True
        except Exception:
            return False
    return True


__all__ = [
    "trigger_lock",
    "is_lock_active",
    "read_lock_data",
    "clear_lock",
    "FLAG_PATH",
    "HISTORY_LOG",
]
