from __future__ import annotations

import logging
import json
import uuid
from typing import Optional

from typing import TYPE_CHECKING

from .bot_registry import BotRegistry
from .data_bot import DataBot
from .coding_bot_interface import self_coding_managed, manager_generate_helper

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .auto_escalation_manager import AutoEscalationManager
    from .bot_database import BotDB
    from .self_coding_manager import SelfCodingManager

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
from context_builder_util import ensure_fresh_weights


registry = BotRegistry()
data_bot = DataBot(start_server=False)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class AutomatedReviewer:
    """Analyse review events and trigger remediation."""

    manager: SelfCodingManager

    def __init__(
        self,
        context_builder: ContextBuilder,
        bot_db: "BotDB" | None = None,
        escalation_manager: "AutoEscalationManager" | None = None,
        *,
        manager: "SelfCodingManager",
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
        self.manager = manager
        try:
            name = getattr(self, "name", getattr(self, "bot_name", self.__class__.__name__))
            self.manager.register_bot(name)
            orch = getattr(self.manager, "evolution_orchestrator", None)
            if orch:
                orch.register_bot(name)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("bot registration failed")
        evo = getattr(self.manager, "evolution_orchestrator", None)
        if evo:
            try:  # pragma: no cover - best effort registration
                evo.register_bot(self.manager.bot_name)
            except Exception:
                self.logger.exception("evolution orchestrator registration failed")
            try:  # pragma: no cover - best effort subscription
                evo._ensure_degradation_subscription()
            except Exception:
                self.logger.exception("failed to subscribe to degradation events")
        if context_builder is None:
            raise ValueError("context_builder is required")
        if not hasattr(context_builder, "build"):
            raise TypeError("context_builder must implement build()")
        self.context_builder = context_builder
        try:
            ensure_fresh_weights(self.context_builder)
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
            vectors: list[tuple[str, str, float]] = []
            try:
                ctx_res = self.context_builder.build(
                    json.dumps({"bot_id": bot_id, "severity": severity}),
                    session_id=session_id,
                    include_vectors=True,
                )
                context_obj = ctx_res
                if isinstance(ctx_res, tuple):
                    context_obj, session_id, vectors = ctx_res
                if isinstance(context_obj, (FallbackResult, ErrorResult)):
                    ctx = ""
                elif isinstance(context_obj, dict):
                    ctx = json.dumps(compress_snippets(context_obj))
                else:
                    ctx = context_obj  # type: ignore[assignment]
            except Exception:
                ctx = ""
                vectors = []
            try:
                manager_generate_helper(self.manager, f"review for bot {bot_id}")
            except Exception:
                self.logger.exception("helper generation failed")
            try:
                self.escalation_manager.handle(
                    f"review for bot {bot_id}",
                    attachments=[ctx],
                    session_id=session_id,
                    vector_metadata=vectors,
                )
            except Exception:
                self.logger.exception("escalation failed")

            roi = 0.0
            errors = 0.0
            try:
                roi = data_bot.roi(str(bot_id))
            except Exception:
                self.logger.exception("failed to fetch ROI for %s", bot_id)
            try:
                errors = data_bot.average_errors(str(bot_id))
            except Exception:
                self.logger.exception("failed to fetch errors for %s", bot_id)
            try:
                data_bot.record_metrics(str(bot_id), float(roi), float(errors))
            except Exception:
                self.logger.exception("failed to record metrics for %s", bot_id)
            try:
                data_bot.check_degradation(str(bot_id), float(roi), float(errors))
            except Exception:
                self.logger.exception("degradation check failed for %s", bot_id)


__all__ = ["AutomatedReviewer"]
