from __future__ import annotations

"""Background agent that forwards alignment warnings to the security auditor."""

import logging
import threading

from .violation_logger import load_recent_alignment_warnings
from .security_auditor import SecurityAuditor


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


__all__ = ["AlignmentReviewAgent"]

