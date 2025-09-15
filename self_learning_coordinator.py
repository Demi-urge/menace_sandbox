from __future__ import annotations

"""Coordinate incremental training from event bus activity."""

from typing import Callable, List, Optional, Tuple, Awaitable
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime
import threading

from pydantic import BaseModel, ValidationError
from sandbox_settings import SandboxSettings

try:  # pragma: no cover - simplified environments may lack full init helpers
    from .self_improvement.init import FileLock, _atomic_write
except Exception as exc:  # pragma: no cover - fail fast when helpers unavailable
    raise ImportError(
        "self_learning_coordinator requires self_improvement.init._atomic_write "
        "for atomic file operations"
    ) from exc
from .unified_event_bus import EventBus
from .data_bot import MetricsDB
from .neuroplasticity import PathwayRecord, Outcome
from .learning_engine import LearningEngine
from .unified_learning_engine import UnifiedLearningEngine
from .action_learning_engine import ActionLearningEngine
from .evaluation_manager import EvaluationManager
from .error_bot import ErrorBot
from .curriculum_builder import CurriculumBuilder
from .self_improvement.baseline_tracker import BaselineTracker
from .dynamic_path_router import resolve_path


logger = logging.getLogger(__name__)


class SelfLearningCoordinator:
    """Listen for events and incrementally update learning engines."""

    def __init__(
        self,
        event_bus: EventBus,
        *,
        learning_engine: LearningEngine | None = None,
        unified_engine: UnifiedLearningEngine | None = None,
        action_engine: ActionLearningEngine | None = None,
        eval_interval: int | None = None,
        metrics_db: MetricsDB | None = None,
        error_bot: ErrorBot | None = None,
        summary_interval: int | None = None,
        curriculum: CurriculumBuilder | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.learning_engine = learning_engine
        self.unified_engine = unified_engine
        self.action_engine = action_engine
        self.running = False
        try:
            settings = SandboxSettings()
        except Exception:  # pragma: no cover - fallback when settings unavailable
            settings = None
        self.eval_interval = (
            eval_interval
            if eval_interval is not None
            else getattr(settings, "self_learning_eval_interval", 0)
        )
        self.metrics_db = metrics_db
        self.error_bot = error_bot
        self.summary_interval = (
            summary_interval
            if summary_interval is not None
            else getattr(settings, "self_learning_summary_interval", 0)
        )
        self.curriculum_builder = curriculum
        self._success_tracker = BaselineTracker()
        self._lock = threading.Lock()
        self._summary_count = 0
        self._train_count = 0
        self._last_eval_ts: str | None = None
        data_dir = getattr(settings, "sandbox_data_dir", ".") if settings else "."
        data_dir = resolve_path(data_dir)
        self._state_path = Path(data_dir) / "self_learning_state.json"
        self._load_state()
        self.evaluation_manager = EvaluationManager(
            learning_engine, unified_engine, action_engine
        )
        self.best_engine: Optional[object] = None
        self._subs: List[Tuple[str, Callable[[str, object], Awaitable[None]]]] = []

    # --------------------------------------------------------------
    def _load_state(self) -> None:
        lock = FileLock(str(self._state_path) + ".lock")
        try:
            with lock:
                data = json.loads(self._state_path.read_text())
            with self._lock:
                self._train_count = int(data.get("train_count", 0))
                self._summary_count = int(data.get("summary_count", 0))
                self._last_eval_ts = data.get("last_eval_ts")
        except FileNotFoundError:
            logger.info(
                "self-learning state file not found at %s; initializing fresh state",
                self._state_path,
            )
            with self._lock:
                self._train_count = 0
                self._summary_count = 0
                self._last_eval_ts = None
            self._save_state()
        except Exception as exc:
            logger.warning(
                "failed to read self-learning state from %s: %s; starting fresh",
                self._state_path,
                exc,
            )
            with self._lock:
                self._train_count = 0
                self._summary_count = 0
                self._last_eval_ts = None
            self._save_state()

    def _save_state(self) -> None:
        with self._lock:
            payload = {
                "train_count": self._train_count,
                "summary_count": self._summary_count,
                "last_eval_ts": self._last_eval_ts,
            }
            lock = FileLock(str(self._state_path) + ".lock")
            try:
                _atomic_write(self._state_path, json.dumps(payload), lock=lock)
            except Exception as exc:
                logger.warning(
                    "failed to persist self-learning state: %s",
                    exc,
                )

    def _reload_intervals(self) -> None:
        try:
            settings = SandboxSettings()
        except Exception:
            return
        self.eval_interval = getattr(
            settings, "self_learning_eval_interval", self.eval_interval
        )
        self.summary_interval = getattr(
            settings, "self_learning_summary_interval", self.summary_interval
        )

    # --------------------------------------------------------------
    class MemoryEvent(BaseModel):
        key: str | None = None
        data: str | None = None

    class CodeEvent(BaseModel):
        summary: str | None = None
        code: str | None = None
        complexity_score: float | None = 0.0

    class WorkflowEvent(BaseModel):
        workflow: List[str] | None = None
        title: str | None = None
        status: str | None = ""
        workflow_duration: float | None = 0.0
        estimated_profit_per_bot: float | None = 0.0

    class PathwayEvent(BaseModel):
        actions: str | None = None
        inputs: str | None = None
        outputs: str | None = None
        exec_time: float | None = 0.0
        resources: str | None = None
        outcome: str | None = "FAILURE"
        roi: float | None = 0.0
        ts: str | None = None

    class ErrorEvent(BaseModel):
        message: str | None = None
        error_type: str | None = None

    class TelemetryEvent(BaseModel):
        error_type: str | None = None
        stack_trace: str | None = None

    class MetricsEvent(BaseModel):
        bot: str
        cpu: float = 0.0
        memory: float = 0.0
        response_time: float = 0.0
        disk_io: float = 0.0
        net_io: float = 0.0
        errors: int = 0
        revenue: float = 0.0
        expense: float = 0.0
        ts: str | None = None

    class TransactionEvent(BaseModel):
        model_id: str
        amount: float = 0.0
        result: str = ""

    class CurriculumEvent(BaseModel):
        error_type: str

    # --------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        topics = [
            ("memory:new", self._on_memory),
            ("code:new", self._on_code),
            ("pathway:new", self._on_pathway),
            ("workflows:new", self._on_workflow),
            ("errors:new", self._on_error),
            ("telemetry:new", self._on_telemetry),
            ("metrics:new", self._on_metrics),
            ("transactions:new", self._on_transaction),
            ("curriculum:new", self._on_curriculum),
        ]
        self._subs = []
        for topic, cb in topics:
            try:
                self.event_bus.subscribe_async(topic, cb)
                self._subs.append((topic, cb))
            except Exception as exc:
                log = getattr(self.error_bot, "logger", logger)
                log.warning("failed to subscribe %s: %s", topic, exc)

    def stop(self) -> None:
        if not self.running:
            return
        unsub = getattr(self.event_bus, "unsubscribe", None)
        for topic, cb in self._subs:
            if unsub:
                try:
                    unsub(topic, cb)
                except Exception as exc:
                    log = getattr(self.error_bot, "logger", logger)
                    log.warning("failed to unsubscribe %s: %s", topic, exc)
        self._subs = []
        self.running = False

    # --------------------------------------------------------------
    async def _on_memory(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.MemoryEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid memory payload: %s", exc)
            return
        actions = str(ev.key or ev.data or "")
        rec = PathwayRecord(
            actions=actions,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=0.0,
        )
        await self._train_all(rec, source="memory")

    async def _on_code(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.CodeEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid code payload: %s", exc)
            return
        actions = str(ev.summary or ev.code or "")
        rec = PathwayRecord(
            actions=actions,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=float(ev.complexity_score or 0.0),
        )
        await self._train_all(rec, source="code")

    async def _on_workflow(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.WorkflowEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid workflow payload: %s", exc)
            return
        if isinstance(ev.workflow, list):
            actions = "->".join(str(s) for s in ev.workflow)
        else:
            actions = str(ev.workflow or ev.title or "")
        status = (ev.status or "").lower()
        oc = Outcome.FAILURE if "fail" in status or "reject" in status else Outcome.SUCCESS
        rec = PathwayRecord(
            actions=actions,
            inputs="",
            outputs="",
            exec_time=float(ev.workflow_duration or 0.0),
            resources="",
            outcome=oc,
            roi=float(ev.estimated_profit_per_bot or 0.0),
        )
        await self._train_all(rec, source="workflow")

    async def _on_pathway(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.PathwayEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid pathway payload: %s", exc)
            return
        try:
            rec = PathwayRecord(
                actions=ev.actions or "",
                inputs=ev.inputs or "",
                outputs=ev.outputs or "",
                exec_time=float(ev.exec_time or 0.0),
                resources=ev.resources or "",
                outcome=Outcome(ev.outcome or "FAILURE"),
                roi=float(ev.roi or 0.0),
                ts=ev.ts or "",
            )
        except Exception:
            logger.warning("failed to build pathway record from payload")
            return
        await self._train_all(rec, source="pathway")

    async def _on_error(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.ErrorEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid error payload: %s", exc)
            return
        msg = str(ev.message or ev.error_type or "")
        rec = PathwayRecord(
            actions=msg,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.0,
        )
        await self._train_all(rec, source="error")

    async def _on_telemetry(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.TelemetryEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid telemetry payload: %s", exc)
            return
        msg = str(ev.error_type or ev.stack_trace or "")
        rec = PathwayRecord(
            actions=msg,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.0,
        )
        await self._train_all(rec, source="telemetry")

    async def _on_metrics(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.MetricsEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid metrics payload: %s", exc)
            return
        try:
            roi = (ev.revenue - ev.expense) / (ev.expense or 1.0)
        except Exception:
            roi = 0.0
        resources = (
            f"cpu={ev.cpu},mem={ev.memory}",
            f"disk={ev.disk_io},net={ev.net_io}",
        )
        oc = Outcome.FAILURE if int(ev.errors) > 0 else Outcome.SUCCESS
        rec = PathwayRecord(
            actions=str(ev.bot),
            inputs="",
            outputs="",
            exec_time=float(ev.response_time),
            resources=resources,
            outcome=oc,
            roi=roi,
            ts=str(ev.ts or ""),
        )
        await self._train_all(rec, source="metrics")

    async def _on_transaction(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.TransactionEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid transaction payload: %s", exc)
            return
        amount = float(ev.amount)
        oc = Outcome.SUCCESS if "success" in ev.result.lower() else Outcome.FAILURE
        rec = PathwayRecord(
            actions=str(ev.model_id),
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=oc,
            roi=amount,
        )
        await self._train_all(rec, source="transaction")

    async def _on_curriculum(self, topic: str, payload: object) -> None:
        if not self.running or not isinstance(payload, dict):
            return
        try:
            ev = self.CurriculumEvent.model_validate(payload)
        except ValidationError as exc:
            logger.warning("invalid curriculum payload: %s", exc)
            return
        err = str(ev.error_type)
        if not err:
            return
        rec = PathwayRecord(
            actions=err,
            inputs="",
            outputs="",
            exec_time=0.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.0,
        )
        await self._train_all(rec, source="curriculum")

    # --------------------------------------------------------------
    async def _train_all(self, rec: PathwayRecord, *, source: str = "unknown") -> None:
        await self._train_record(rec)
        self._reload_intervals()
        do_eval = False
        do_summary = False
        with self._lock:
            self._train_count += 1
            train_count = self._train_count
            if self.eval_interval and self._train_count >= self.eval_interval:
                do_eval = True
                self._train_count = 0
                self._last_eval_ts = datetime.utcnow().isoformat()
            if self.summary_interval and self.error_bot:
                self._summary_count += 1
                if self._summary_count >= self.summary_interval:
                    do_summary = True
                    self._summary_count = 0
        logger.info(
            "processed training record %s from %s (count=%d)",
            getattr(rec, "ts", ""),
            source,
            train_count,
        )
        if self.metrics_db:
            try:
                self.metrics_db.log_training_stat(source, True)
            except Exception as exc:
                logger.exception("metrics_db failed during log_training_stat: %s", exc)
        if do_eval:
            self._evaluate_all()
        if do_summary:
            await self._train_from_summary()
        self._save_state()

    async def _train_record(self, rec: PathwayRecord) -> None:
        tasks: set[asyncio.Task] = set()
        meta: dict[asyncio.Task, tuple[object, str, int]] = {}

        def schedule(engine: object, name: str, attempt: int = 0) -> None:
            task = asyncio.create_task(self._partial_train(engine, rec))
            meta[task] = (engine, name, attempt)
            tasks.add(task)

        if self.learning_engine:
            schedule(self.learning_engine, "learning_engine")
        if self.unified_engine:
            schedule(self.unified_engine, "unified_engine")
        if self.action_engine:
            schedule(self.action_engine, "action_engine")

        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = pending
            for t in done:
                engine, name, attempt = meta.pop(t)
                try:
                    t.result()
                except Exception as exc:
                    logger.exception("%s failed during partial_train: %s", name, exc)
                    if attempt < 1:
                        schedule(engine, name, attempt + 1)

    async def _partial_train(self, engine: object, rec: PathwayRecord) -> None:
        await asyncio.to_thread(engine.partial_train, rec)

    async def _train_from_summary(self) -> None:
        if self.curriculum_builder:
            items = self.curriculum_builder.publish()
            logger.info(
                "published %d curriculum entries via CurriculumBuilder",
                len(items),
            )
            return
        if not self.error_bot:
            return
        summary = self.error_bot.summarize_telemetry()
        logger.info(
            "publishing %d summary records from ErrorBot",
            len(summary),
        )
        summary.sort(key=lambda s: s.get("success_rate", 0.0))
        for idx, item in enumerate(summary, start=1):
            rate = float(item.get("success_rate", 0.0))
            avg = self._success_tracker.get("success_rate")
            delta = rate - avg
            self._success_tracker.update(
                success_rate=rate, success_rate_delta=delta
            )
            history = self._success_tracker.delta_history("success_rate")
            streak = 0
            for d in reversed(history):
                if d <= 0:
                    streak += 1
                else:
                    break
            outcome = (
                Outcome.FAILURE if delta < 0 and streak >= 3 else Outcome.SUCCESS
            )
            rec = PathwayRecord(
                actions=str(item.get("error_type", "")),
                inputs="",
                outputs="",
                exec_time=0.0,
                resources="",
                outcome=outcome,
                roi=0.0,
            )
            await self._train_record(rec)
            logger.info(
                "processed summary training record %s (%d/%d) baseline=%.3f delta=%.3f streak=%d",
                getattr(rec, "ts", ""),
                idx,
                len(summary),
                avg,
                delta,
                streak,
            )

    def _evaluate_all(self) -> None:
        try:
            self.evaluation_manager.evaluate_all()
            self.best_engine = self.evaluation_manager.best_engine()
        except Exception as exc:
            logger.exception(
                "evaluation_manager failed during evaluate_all: %s", exc
            )


__all__ = ["SelfLearningCoordinator", "CurriculumBuilder"]
