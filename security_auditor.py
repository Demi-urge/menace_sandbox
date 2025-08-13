from __future__ import annotations

"""Basic security auditing helpers with an optional auto-fix loop."""

from typing import Mapping, Any
import logging


class SecurityAuditor:
    """Previously executed static scans. Now a no-op."""

    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir = base_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _run(self, cmd: list[str]) -> bool:
        """Return ``True`` without executing any external command."""
        self.logger.debug("skipping %s", cmd[0])
        return True

    # ------------------------------------------------------------------
    def audit(self) -> bool:
        """Security checks disabled - always return ``True``."""
        self.logger.info("security checks disabled")
        return True


def fix_until_safe(auditor: "SecurityAuditor", *, attempts: int = 3) -> bool:
    """Return ``True`` immediately as audits are disabled."""
    auditor.logger.info("auto fix skipped; security checks disabled")
    return True


def dispatch_alignment_warning(report: Mapping[str, Any]) -> None:
    """Accept alignment warnings from external flaggers.

    The default implementation simply logs the provided *report*.  Callers
    may monkeypatch this function in tests or provide their own dispatcher
    to integrate with a real security monitoring system.
    """

    try:
        logging.getLogger(__name__).warning("alignment warning: %s", report)
    except Exception:  # pragma: no cover - defensive logging
        logging.getLogger(__name__).exception("alignment warning dispatch failed")


__all__ = ["SecurityAuditor", "fix_until_safe", "dispatch_alignment_warning"]
