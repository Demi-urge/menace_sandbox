from __future__ import annotations

"""Background agent that forwards alignment warnings to the security auditor."""

import argparse
import logging
import threading
from typing import Any, List, Dict

from .violation_logger import (
    load_persisted_alignment_warnings,
    mark_warning_approved,
    mark_warning_pending,
    mark_warning_rejected,
    update_warning_status,
)
from .security_auditor import SecurityAuditor


def load_recent_alignment_warnings() -> List[Dict[str, Any]]:
    """Return pending alignment warnings (compatibility wrapper)."""

    return load_persisted_alignment_warnings(review_status="pending")


class AlignmentReviewAgent:
    """Poll alignment warnings and hand them off to :class:`SecurityAuditor`."""

    def __init__(
        self,
        interval: float = 5.0,
        auditor: SecurityAuditor | None = None,
    ) -> None:
        self.interval = float(interval)
        self.auditor = auditor or SecurityAuditor()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: set[str] = set()

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background polling loop."""

        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop the background polling loop."""

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                warnings = load_recent_alignment_warnings()
                for rec in warnings:
                    entry_id = str(rec.get("entry_id"))
                    if entry_id in self._seen:
                        continue
                    self._seen.add(entry_id)
                    try:
                        self.auditor.audit(rec)
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("audit failed for %s", entry_id)
            except Exception:  # pragma: no cover - defensive
                self.logger.exception("alignment review loop failed")
            self._stop.wait(self.interval)

    # ------------------------------------------------------------------
    def set_review_status(self, entry_id: str, status: str) -> None:
        """Update the review status for a warning."""

        update_warning_status(entry_id, status)


def review_warning(entry_id: str, status: str) -> None:
    """Utility for marking the review decision of a warning."""

    if status == "approved":
        mark_warning_approved(entry_id)
    elif status == "rejected":
        mark_warning_rejected(entry_id)
    else:
        mark_warning_pending(entry_id)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Record review decision for an alignment warning"
    )
    parser.add_argument("entry_id")
    parser.add_argument("status", choices=["pending", "approved", "rejected"])
    args = parser.parse_args()
    review_warning(args.entry_id, args.status)


if __name__ == "__main__":  # pragma: no cover - CLI utility
    _cli()


__all__ = ["AlignmentReviewAgent", "review_warning", "load_recent_alignment_warnings"]

