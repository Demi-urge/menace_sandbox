"""Basic compliance checks for high risk actions."""

from __future__ import annotations

import logging
import os
from typing import Dict, Mapping

from .audit_trail import AuditTrail
from .roles import load_role_permissions

# Default limit used when ``MAX_TRADE_VOLUME`` is unset
DEFAULT_MAX_TRADE_VOLUME = 1000.0


class ComplianceChecker:
    """Validate trades and permissions before execution."""

    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = log_path or os.getenv("COMPLIANCE_LOG", "compliance.log")
        self.audit = AuditTrail(self.log_path)
        self.permissions = load_role_permissions()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def check_trade(self, trade: Mapping[str, object]) -> bool:
        """Return ``True`` if ``trade`` respects configured limits."""
        if not isinstance(trade, Mapping):
            raise TypeError("trade must be a mapping")
        volume_raw = trade.get("volume", 0)
        try:
            volume = float(volume_raw)
        except Exception as exc:
            raise TypeError("trade volume must be numeric") from exc
        limit = float(os.getenv("MAX_TRADE_VOLUME", str(DEFAULT_MAX_TRADE_VOLUME)))
        if volume > limit:
            self._log_event({"type": "volume", "trade": dict(trade), "allowed": False})
            return False
        return True

    # ------------------------------------------------------------------
    def _log_event(self, payload: Dict[str, object]) -> None:
        """Write *payload* to the audit trail."""
        try:
            self.audit.record(payload)
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("audit log failure: %s", exc)

    # ------------------------------------------------------------------
    def verify_permission(self, role: str, action: str) -> bool:
        """Return ``True`` when ``role`` is allowed to perform ``action``."""
        if not isinstance(role, str) or not isinstance(action, str):
            raise TypeError("role and action must be strings")
        role_lc = role.lower()
        perms = self.permissions.get(role_lc, set())
        ok = action in perms
        self._log_event({"type": "permission", "role": role_lc, "action": action, "allowed": ok})
        return ok

