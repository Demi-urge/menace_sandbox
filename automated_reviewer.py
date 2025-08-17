from __future__ import annotations

import logging
import json
from typing import Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .auto_escalation_manager import AutoEscalationManager
from vector_service import ContextBuilder
try:  # pragma: no cover - graceful fallback when dependency missing
    from vector_service import ErrorResult
except Exception:  # pragma: no cover - compatibility
    class ErrorResult(Exception):
        """Fallback ErrorResult when vector service dependency missing."""
        pass


class AutomatedReviewer:
    """Analyse review events and trigger remediation."""

    def __init__(
        self,
        bot_db: "BotDB" | None = None,
        escalation_manager: "AutoEscalationManager" | None = None,
    ) -> None:
        if bot_db is None:
            from .bot_database import BotDB

            bot_db = BotDB()
        self.bot_db = bot_db
        if escalation_manager is None:
            from .auto_escalation_manager import AutoEscalationManager

            escalation_manager = AutoEscalationManager()
        self.escalation_manager = escalation_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def handle(self, event: object) -> None:
        """Process a review event."""
        bot_id: Optional[str] = None
        severity: Optional[str] = None
        if isinstance(event, dict):
            bot_id = event.get("bot_id")
            severity = event.get("severity")
        if not bot_id:
            self.logger.warning("invalid review event: %s", event)
            return
        if severity == "critical":
            try:
                self.bot_db.update_bot(int(bot_id), status="disabled")
            except Exception:
                self.logger.exception("failed disabling bot %s", bot_id)
            try:
                builder = ContextBuilder()
                ctx = builder.build(json.dumps({"bot_id": bot_id, "severity": severity}))
                if isinstance(ctx, ErrorResult):
                    self.logger.error("context build failed: %s", ctx)
                    ctx = ""
                self.escalation_manager.handle(
                    f"review for bot {bot_id}", attachments=[ctx]
                )
            except Exception:
                self.logger.exception("escalation failed")


__all__ = ["AutomatedReviewer"]

