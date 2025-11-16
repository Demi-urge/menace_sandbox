from __future__ import annotations

"""Persistent review queue for workflows requiring human review or shadow testing.

Entries are stored as newline-delimited JSON objects so other automation can
inspect the queue without needing in-memory state."""

from pathlib import Path
from datetime import datetime
import json
from typing import Any

QUEUE_FILE = Path("review_queue.jsonl")


def enqueue_for_review(workflow_id: str) -> None:
    """Append *workflow_id* to the persistent review queue."""
    entry = {"workflow_id": workflow_id, "timestamp": datetime.utcnow().isoformat()}
    with QUEUE_FILE.open("a", encoding="utf-8") as fh:
        json.dump(entry, fh)
        fh.write("\n")


def in_review_queue(workflow_id: str) -> bool:
    """Return True if *workflow_id* is queued for manual review."""
    if not QUEUE_FILE.exists():
        return False
    with QUEUE_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("workflow_id") == workflow_id:
                return True
    return False


def should_bypass_auto_demotion(workflow_id: str) -> bool:
    """Return True if automated demotion should be bypassed for *workflow_id*.

    Downstream automation can use this helper to avoid demoting workflows that
    are awaiting manual review."""
    return in_review_queue(workflow_id)
