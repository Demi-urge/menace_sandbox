from __future__ import annotations

import logging
import json
import uuid
from typing import Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .auto_escalation_manager import AutoEscalationManager
    from .bot_database import BotDB

# ``vector_service`` is required.  Fail fast if the import is unavailable.
try:  # pragma: no cover - optional dependency used in runtime
    from vector_service import CognitionLayer, ContextBuilder
except Exception as exc:  # pragma: no cover - dependency missing or broken
    raise RuntimeError("vector_service import failed") from exc

# ``FallbackResult`` and ``ErrorResult`` may not always be present on the
# ``vector_service`` module.  Provide light-weight fallbacks so our type checks
# remain functional during tests when the real classes are absent.
try:  # pragma: no cover - optional dependency used in runtime
    from vector_service import FallbackResult  # type: ignore
except Exception:  # pragma: no cover - module missing the attribute
    class FallbackResult:  # type: ignore[override]
        pass

try:  # pragma: no cover - optional dependency used in runtime
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - module missing the attribute
    class ErrorResult(Exception):  # type: ignore[override]
        pass

from snippet_compressor import compress_snippets


class AutomatedReviewer:
    """Analyse review events and trigger remediation."""

    def __init__(
        self,
        context_builder: ContextBuilder,
        bot_db: "BotDB" | None = None,
        escalation_manager: "AutoEscalationManager" | None = None,
    ) -> None:
        if bot_db is None:
            from .bot_database import BotDB

            bot_db = BotDB()
        self.bot_db = bot_db
        if escalation_manager is None:
            from .auto_escalation_manager import AutoEscalationManager

            escalation_manager = AutoEscalationManager(context_builder=context_builder)
        self.escalation_manager = escalation_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        if context_builder is None:
            raise ValueError("context_builder is required")
        if not hasattr(context_builder, "build"):
            raise TypeError("context_builder must implement build()")
        self.context_builder = context_builder
        try:
            self.context_builder.refresh_db_weights()
        except Exception as exc:
            self.logger.error("context builder refresh failed: %s", exc)
            raise RuntimeError("context builder refresh failed") from exc
        try:
            self.cognition_layer = CognitionLayer(context_builder=context_builder)
        except Exception:  # pragma: no cover - optional dependency failed
            self.logger.error("failed to initialise CognitionLayer", exc_info=True)
            raise

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

            session_id = str(uuid.uuid4())
            ctx: str = ""
            try:
                result = self.context_builder.build(
                    json.dumps({"bot_id": bot_id, "severity": severity}),
                    session_id=session_id,
                )
                if isinstance(result, (FallbackResult, ErrorResult)):
                    ctx = ""
                elif isinstance(result, dict):
                    ctx = json.dumps(compress_snippets(result))
                else:
                    ctx = result  # type: ignore[assignment]
            except Exception:
                ctx = ""
            try:
                self.escalation_manager.handle(
                    f"review for bot {bot_id}",
                    attachments=[ctx],
                    session_id=session_id,
                )
            except Exception:
                self.logger.exception("escalation failed")


__all__ = ["AutomatedReviewer"]
