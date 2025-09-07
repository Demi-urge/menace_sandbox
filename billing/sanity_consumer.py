from __future__ import annotations

"""Consumer for billing anomaly events.

This module subscribes to the ``"billing.anomaly"`` topic of
:class:`~unified_event_bus.UnifiedEventBus`.  Each event triggers a feedback
cycle that either applies a self‑generated patch or tweaks generation
parameters on the :class:`~self_coding_engine.SelfCodingEngine`.

Results are logged via :func:`menace_sanity_layer.record_event` and recorded in
:mod:`discrepancy_db` for later analysis.
"""

import logging
from typing import Any, Dict, TYPE_CHECKING

from unified_event_bus import UnifiedEventBus
from sanity_feedback import SanityFeedback
from self_coding_engine import SelfCodingEngine

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from vector_service.context_builder import ContextBuilder
import menace_sanity_layer
from dynamic_path_router import resolve_path

try:  # pragma: no cover - optional dependency
    from code_database import CodeDB
except Exception:  # pragma: no cover - best effort
    CodeDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from menace_memory_manager import MenaceMemoryManager
except Exception:  # pragma: no cover - best effort
    MenaceMemoryManager = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from discrepancy_db import DiscrepancyDB, DiscrepancyRecord
except Exception:  # pragma: no cover - best effort
    DiscrepancyDB = None  # type: ignore
    DiscrepancyRecord = None  # type: ignore

logger = logging.getLogger(__name__)


class SanityConsumer:
    """Subscribe to billing anomaly events and trigger self-correction."""

    def __init__(
        self,
        event_bus: UnifiedEventBus | None = None,
        *,
        engine: SelfCodingEngine | None = None,
        context_builder: "ContextBuilder" | None = None,
    ) -> None:
        self.event_bus = event_bus or getattr(
            menace_sanity_layer, "_EVENT_BUS", UnifiedEventBus()
        )
        self.event_bus.subscribe("billing.anomaly", self._handle)
        self._engine: SelfCodingEngine | None = engine
        self._feedback: SanityFeedback | None = None
        self._outcome_db = DiscrepancyDB() if DiscrepancyDB is not None else None
        self._context_builder = context_builder

    # ------------------------------------------------------------------
    def _get_engine(self) -> SelfCodingEngine:
        if self._engine is None:
            try:
                try:
                    code_db = CodeDB() if CodeDB is not None else object()
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed to initialise CodeDB")
                    code_db = object()
                try:
                    memory_mgr = (
                        MenaceMemoryManager() if MenaceMemoryManager is not None else object()
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed to initialise MenaceMemoryManager")
                    memory_mgr = object()

                builder = self._context_builder
                if builder is None:
                    raise RuntimeError("ContextBuilder required")
                builder.refresh_db_weights()
                self._engine = SelfCodingEngine(
                    code_db, memory_mgr, context_builder=builder
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to initialise SelfCodingEngine")
                raise
        return self._engine

    def _get_feedback(self) -> SanityFeedback:
        if self._feedback is None:
            self._feedback = SanityFeedback(
                self._get_engine(), outcome_db=self._outcome_db
            )
            if self._context_builder is not None:
                # Share builder so feedback analysers can inspect engine context
                setattr(self._feedback, "context_builder", self._context_builder)
        return self._feedback

    # ------------------------------------------------------------------
    def _handle(self, _topic: str, event: Dict[str, Any]) -> None:
        """Handle a published billing anomaly event."""

        feedback = self._get_feedback()
        engine = feedback.engine
        meta = event.get("metadata", {}) if isinstance(event, dict) else {}
        event_type = (
            event.get("event_type", "billing_anomaly")
            if isinstance(event, dict)
            else "billing_anomaly"
        )

        patch_id = None
        success = False
        try:
            path = meta.get("module") or meta.get("path")
            if path:
                try:
                    target = resolve_path(path if path.endswith(".py") else f"{path}.py")
                    patch_id, success, _ = engine.apply_patch(
                        target,
                        f"address {event_type} anomaly",
                        reason=event_type,
                        trigger="billing_anomaly",
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("patch application failed", extra={"path": path})
            else:
                update_fn = getattr(engine, "update_generation_params", None)
                if update_fn is not None:
                    try:
                        changes = update_fn(meta) or {}
                        if changes:
                            logger.info(
                                "generation params updated", extra={"changes": changes}
                            )
                    except Exception:  # pragma: no cover - best effort
                        logger.exception("generation parameter update failed")
        finally:
            log_meta = {**meta, "patch_id": patch_id, "success": success}
            try:  # log feedback for future optimisation
                menace_sanity_layer.record_event(
                    event_type, log_meta, self_coding_engine=engine
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to record sanity event")

            if self._outcome_db is not None and DiscrepancyRecord is not None:
                try:
                    rec = DiscrepancyRecord(message=event_type, metadata=log_meta)
                    self._outcome_db.add(rec)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("discrepancy logging failed")


__all__ = ["SanityConsumer"]
