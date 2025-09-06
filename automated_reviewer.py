from __future__ import annotations

import logging
import json
from typing import Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .auto_escalation_manager import AutoEscalationManager
    from .bot_database import BotDB
# Optional dependency: vector_service
try:  # pragma: no cover - optional dependency used in runtime
    from vector_service import CognitionLayer, ContextBuilder
except Exception:  # pragma: no cover - fallback when dependency missing
    CognitionLayer = ContextBuilder = None  # type: ignore
    logging.getLogger(__name__).warning(
        "vector_service import failed; cognition features disabled"
    )


class AutomatedReviewer:
    """Analyse review events and trigger remediation."""

    def __init__(
        self,
        bot_db: "BotDB" | None = None,
        escalation_manager: "AutoEscalationManager" | None = None,
        context_builder: "ContextBuilder" | None = None,
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
        if CognitionLayer is not None and ContextBuilder is not None:
            try:
                builder = context_builder or ContextBuilder()
                self.cognition_layer = CognitionLayer(context_builder=builder)
            except Exception:  # pragma: no cover - optional dependency failed
                self.cognition_layer = None
                self.logger.warning("failed to initialise CognitionLayer", exc_info=True)
        else:  # pragma: no cover - dependency missing at import time
            self.cognition_layer = None
            self.logger.warning(
                "CognitionLayer unavailable due to missing vector_service dependency"
            )

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
            ctx = ""
            if self.cognition_layer is not None:
                try:
                    ctx, _sid = self.cognition_layer.query(
                        json.dumps({"bot_id": bot_id, "severity": severity})
                    )
                except Exception:
                    ctx = ""
            try:
                self.escalation_manager.handle(
                    f"review for bot {bot_id}", attachments=[ctx]
                )
            except Exception:
                self.logger.exception("escalation failed")


__all__ = ["AutomatedReviewer"]
