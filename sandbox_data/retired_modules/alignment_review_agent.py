from __future__ import annotations

"""Fallback alignment review agent for sandbox execution.

The real Menace deployment relies on cloud-hosted governance services. When we
run locally we still want a functional polling loop that surfaces any
alignment warnings recorded in :mod:`violation_logger`. The implementation
below keeps the public API compatible while avoiding heavyweight dependencies.
"""

import argparse
import logging
import threading
from typing import Any, Dict, List

try:  # pragma: no cover - prefer package relative imports
    from menace_sandbox.violation_logger import (
        load_persisted_alignment_warnings,
        mark_warning_approved,
        mark_warning_pending,
        mark_warning_rejected,
        update_warning_status,
    )
except ImportError:  # pragma: no cover - flat layout fallback
    from violation_logger import (  # type: ignore
        load_persisted_alignment_warnings,
        mark_warning_approved,
        mark_warning_pending,
        mark_warning_rejected,
        update_warning_status,
    )

try:  # pragma: no cover - prefer package relative imports
    from menace_sandbox.security_auditor import SecurityAuditor
except ImportError:  # pragma: no cover - flat layout fallback
    from security_auditor import SecurityAuditor  # type: ignore

__all__ = [
    "AlignmentReviewAgent",
    "review_warning",
    "load_recent_alignment_warnings",
    "summarize_warnings",
]


def load_recent_alignment_warnings() -> List[Dict[str, Any]]:
    """Return warnings awaiting manual review."""

    return load_persisted_alignment_warnings(review_status="pending")


def summarize_warnings() -> Dict[str, Any]:
    """Return a compact summary of pending warnings grouped by severity."""

    warnings = load_persisted_alignment_warnings(review_status="pending")
    severity_counts: Dict[int, int] = {}
    modules: set[str] = set()
    for record in warnings:
        severity = int(record.get("severity", 0))
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        module_hint = record.get("patch_link") or record.get("violation_type")
        if module_hint:
            modules.add(str(module_hint))
    return {"counts": severity_counts, "modules": sorted(modules)}


class AlignmentReviewAgent:
    """Poll :mod:`violation_logger` for new warnings and invoke a security auditor."""

    def __init__(
        self,
        interval: float = 5.0,
        auditor: SecurityAuditor | None = None,
    ) -> None:
        self.interval = float(interval)
        self.auditor = auditor or SecurityAuditor()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: set[str] = set()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="alignment-review", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                summary = summarize_warnings()
                try:
                    self.auditor.audit({"summary": summary})
                except Exception:  # pragma: no cover - downstream failures
                    self.logger.exception("alignment summary audit failed")
                for record in load_recent_alignment_warnings():
                    entry_id = str(record.get("entry_id"))
                    if entry_id in self._seen:
                        continue
                    self._seen.add(entry_id)
                    try:
                        self.auditor.audit(record)
                    except Exception:  # pragma: no cover - downstream failures
                        self.logger.exception("alignment audit failed for %s", entry_id)
            except Exception:  # pragma: no cover - defensive
                self.logger.exception("alignment review loop crashed")
            self._stop_event.wait(self.interval)

    # ------------------------------------------------------------------
    def set_review_status(self, entry_id: str, status: str) -> None:
        update_warning_status(entry_id, status)


def review_warning(entry_id: str, status: str) -> None:
    status = status.lower().strip()
    if status == "approved":
        mark_warning_approved(entry_id)
    elif status == "rejected":
        mark_warning_rejected(entry_id)
    else:
        mark_warning_pending(entry_id)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Record review decision for an alignment warning",
    )
    parser.add_argument("entry_id", nargs="?")
    parser.add_argument(
        "status", nargs="?", choices=["pending", "approved", "rejected"]
    )
    parser.add_argument("--summary", action="store_true", help="print pending warning summary")
    args = parser.parse_args()

    if args.summary:
        print(summarize_warnings())
        return

    if not args.entry_id or not args.status:
        parser.error("entry_id and status required unless --summary is used")
    review_warning(args.entry_id, args.status)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _cli()

