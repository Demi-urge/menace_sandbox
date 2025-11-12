from __future__ import annotations

"""Background service running telemetry-driven debugging."""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

from pathlib import Path
from .telemetry_feedback import TelemetryFeedback
from .error_logger import ErrorLogger
from .self_coding_engine import SelfCodingEngine
try:  # pragma: no cover - fail fast if self-coding manager missing
    from .self_coding_manager import SelfCodingManager, internalize_coding_bot
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError(
        "DebugLoopService requires SelfCodingManager; install self-coding dependencies."
    ) from exc
from .model_automation_pipeline import ModelAutomationPipeline
from .unified_event_bus import UnifiedEventBus
from .code_database import CodeDB
from .menace_memory_manager import MenaceMemoryManager
from .knowledge_graph import KnowledgeGraph
from .bot_registry import BotRegistry
from .data_bot import DataBot, persist_sc_thresholds
from .self_coding_thresholds import get_thresholds
from .coding_bot_interface import prepare_pipeline_for_bootstrap

try:  # pragma: no cover - optional vector service dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback when dependency missing
    from vector_service.context_builder import ContextBuilder  # type: ignore


class DebugLoopService:
    """Run :class:`TelemetryFeedback` continuously and archive crash logs."""

    def __init__(
        self,
        feedback: TelemetryFeedback | None = None,
        *,
        graph: KnowledgeGraph | None = None,
        context_builder: ContextBuilder,
        bot_registry: BotRegistry | None = None,
        data_bot: DataBot | None = None,
    ) -> None:
        """Create service.

        Parameters
        ----------
        context_builder:
            Preconfigured :class:`~vector_service.ContextBuilder` used when creating
            telemetry and self-coding components.
        """
        self.graph = graph or KnowledgeGraph()
        if feedback is None:
            try:
                context_builder.refresh_db_weights()
            except Exception:
                pass
            logger = ErrorLogger(
                knowledge_graph=self.graph, context_builder=context_builder
            )
            engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=context_builder
            )
            bus = UnifiedEventBus()
            registry = bot_registry or BotRegistry(event_bus=bus)
            data_bot = data_bot or DataBot()
            pipeline, promote_pipeline = prepare_pipeline_for_bootstrap(
                pipeline_cls=ModelAutomationPipeline,
                context_builder=context_builder,
                bot_registry=registry,
                data_bot=data_bot,
                event_bus=bus,
            )
            _th = get_thresholds("DebugLoopService")
            persist_sc_thresholds(
                "DebugLoopService",
                roi_drop=_th.roi_drop,
                error_increase=_th.error_increase,
                test_failure_increase=_th.test_failure_increase,
            )
            manager = internalize_coding_bot(
                "DebugLoopService",
                engine,
                pipeline,
                data_bot=data_bot,
                bot_registry=registry,
                event_bus=bus,
                roi_threshold=_th.roi_drop,
                error_threshold=_th.error_increase,
                test_failure_threshold=_th.test_failure_increase,
            )
            if not isinstance(manager, SelfCodingManager):  # pragma: no cover - safety
                raise RuntimeError(
                    "internalize_coding_bot failed to return a SelfCodingManager"
                )
            promote_pipeline(manager)
            feedback = TelemetryFeedback(
                logger,
                manager,
            )
        self.feedback = feedback
        self.logger = logging.getLogger(self.__class__.__name__)
        self.failure_count = 0
        self._thread: Optional[threading.Thread] = None
        self._stop: Optional[threading.Event] = None

    # ------------------------------------------------------------------
    def collect_crash_traces(self, log_dir: str = ".") -> None:
        """Scan log files for crash traces and store them."""
        for log in Path(log_dir).glob("*.log"):
            if log.name in {"supervisor.log", "restart.log"}:
                continue
            try:
                text = log.read_text(encoding="utf-8")
            except Exception:
                continue
            if "Traceback" not in text:
                continue
            trace = "\n".join(text.splitlines()[-20:])
            self.graph.add_crash_trace(log.stem, trace)

    def run_continuous(
        self,
        interval: float = 300.0,
        *,
        stop_event: threading.Event | None = None,
        log_dir: str = ".",
    ) -> None:
        """Start the feedback loop in a thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop = stop_event or threading.Event()
        self.feedback.interval = int(interval)

        def _loop() -> None:
            self.feedback.start()
            try:
                while not self._stop.is_set():
                    time.sleep(interval)
                    try:
                        self.collect_crash_traces(log_dir)
                    except Exception as exc:
                        self.logger.exception("collect_crash_traces failed: %s", exc)
                        self.failure_count += 1
            finally:
                self.feedback.stop()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()


__all__ = ["DebugLoopService"]
