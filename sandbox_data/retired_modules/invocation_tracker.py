from __future__ import annotations

"""Invocation tracker for Security AI oversight enforcement."""

from typing import Any, List
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dynamic_path_router import resolve_path

from logging_utils import get_logger

# Path for Security AI invocation logs
INVOCATION_LOG = Path(
    os.environ.get("INVOCATION_LOG", resolve_path("logs/security_ai_invocations.jsonl"))
)

logger = get_logger(__name__)


def log_invocation(timestamp: float, action_id: str) -> None:
    """Append a Security AI invocation record.

    Parameters
    ----------
    timestamp: float
        Time the reward system was called.
    action_id: str
        Identifier of the Menace action being evaluated.
    """
    INVOCATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": timestamp, "action_id": action_id}
    try:
        with INVOCATION_LOG.open("a", encoding="utf-8") as fh:
            json.dump(entry, fh)
            fh.write("\n")
    except Exception as exc:
        logger.error("log_invocation failed: %s", exc)
        tmp_path = INVOCATION_LOG.with_name(INVOCATION_LOG.name + ".tmp")
        try:
            with tmp_path.open("a", encoding="utf-8") as fh:
                json.dump(entry, fh)
                fh.write("\n")
        except Exception as exc2:
            logger.error("tmp log failed: %s", exc2)


def load_recent_invocations(window_seconds: int = 300) -> List[str]:
    """Return action ids logged within the last ``window_seconds``."""
    cutoff = time.time() - window_seconds
    action_ids: List[str] = []
    if not INVOCATION_LOG.exists():
        return action_ids
    try:
        with INVOCATION_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    ts = float(entry.get("timestamp", 0))
                except Exception:
                    continue
                if ts >= cutoff:
                    aid = entry.get("action_id")
                    if isinstance(aid, str):
                        action_ids.append(aid)
    except Exception as exc:
        logger.error("reading invocation log failed: %s", exc)
    return action_ids


def _load_action_log(path: str | Path) -> List[str]:
    """Return list of action ids recorded in Menace log at *path*."""
    p = Path(resolve_path(str(path)))
    ids: List[str] = []
    if not p.exists():
        return ids
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = entry.get("action_id") or entry.get("id")
                if isinstance(aid, str):
                    ids.append(aid)
    except Exception as exc:
        logger.error("failed to load actions from %s: %s", p, exc)
    return ids


def detect_missing_invocations(
    menace_action_log: str | Path | None = None,
    invocation_log: Path | str = INVOCATION_LOG,
) -> List[str]:
    """Compare Menace actions with Security AI invocations.

    Returns a list of action ids that were executed by Menace but never
    processed by Security AI.
    """
    action_log = menace_action_log or os.environ.get(
        "MENACE_ACTION_LOG", "/mnt/shared/menace_logs/actions.jsonl"
    )
    action_ids = set(_load_action_log(action_log))
    invoked = set(_load_action_log(invocation_log))
    missing = sorted(action_ids - invoked)
    return missing


def flag_avoidance_events(missing_ids: List[str]) -> dict[str, Any]:
    """Return summary dict for potential invocation avoidance."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    severity = len(missing_ids)
    events = [
        {"action_id": aid, "detected_at": timestamp, "severity": severity}
        for aid in missing_ids
    ]
    return {
        "detected_at": timestamp,
        "missing": len(missing_ids),
        "events": events,
        "severity_score": severity,
    }


__all__ = [
    "log_invocation",
    "load_recent_invocations",
    "detect_missing_invocations",
    "flag_avoidance_events",
    "INVOCATION_LOG",
]
