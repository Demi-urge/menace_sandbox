from __future__ import annotations

"""Periodic self-hosted model evaluation and deployment."""

import logging
import time
from threading import Event
import threading
import os
import asyncio

from .evaluation_manager import EvaluationManager
from .cross_model_comparator import CrossModelComparator
from .neuroplasticity import PathwayDB
from .evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
from .workflow_cloner import WorkflowCloner
from .evolution_history_db import EvolutionEvent
from .unified_event_bus import UnifiedEventBus
from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler, _AsyncScheduler
from typing import TYPE_CHECKING

from .retry_utils import publish_with_retry
from .error_flags import RAISE_ERRORS

if TYPE_CHECKING:  # pragma: no cover - avoid heavy import
    from .watchdog import Watchdog


class _SimpleScheduler:
    """Minimal in-process scheduler used when APScheduler is unavailable."""

    def __init__(self) -> None:
        self.tasks: list[tuple[float, callable, str]] = []
        self.stop = Event()
        self.thread: threading.Thread | None = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_job(self, func: callable, interval: float, id: str) -> None:
        self.tasks.append((interval, func, id))
        if not self.thread:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self) -> None:
        next_call = time.time()
        while not self.stop.is_set():
            now = time.time()
            if now >= next_call:
                for interval, fn, jid in self.tasks:
                    try:
                        fn()
                    except BaseException:
                        if not self.stop.is_set():
                            self.logger.exception("job %s failed", jid)
                        else:
                            raise
                next_call = now + min(i for i, _, _ in self.tasks)
            time.sleep(0.1)

    def shutdown(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=0)


try:  # pragma: no cover - optional dependency
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - APScheduler missing
    BackgroundScheduler = None  # type: ignore


class ModelEvaluationService:
    """Evaluate engines, redeploy best model and clone workflows."""

    def __init__(
        self,
        manager: EvaluationManager | None = None,
        comparator: CrossModelComparator | None = None,
        cloner: WorkflowCloner | None = None,
        *,
        event_bus: UnifiedEventBus | None = None,
        watchdog: Watchdog | None = None,
    ) -> None:
        self.manager = manager or EvaluationManager()
        self.comparator = comparator or CrossModelComparator(
            PathwayDB(), EvaluationHistoryDB()
        )
        self.cloner = cloner or WorkflowCloner()
        self.event_bus = event_bus
        self.watchdog = watchdog
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: dict[str, dict] = {}
        self._last_event_id: int | None = None
        self.scheduler: object | None = None
        if self.event_bus:
            try:
                self.event_bus.subscribe("evaluation:result", self._on_result)
            except Exception as exc:
                self.logger.exception("subscribe failed: %s", exc)
                if self.watchdog:
                    self.watchdog.escalate(f"evaluation subscribe failed: {exc}")
                if RAISE_ERRORS:
                    raise

    def _on_result(self, topic: str, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        engine = payload.get("engine")
        result = payload.get("result")
        if isinstance(engine, str) and isinstance(result, dict):
            self._results[engine] = result

    def _persist_results(self, results: dict[str, dict]) -> None:
        for name, res in results.items():
            try:
                score = float(res.get("cv_score", 0.0))
            except Exception as exc:
                self.logger.exception("score parse failed for %s: %s", name, exc)
                if self.watchdog:
                    self.watchdog.escalate(f"score parse failed for {name}: {exc}")
                score = 0.0
            try:
                rec = EvaluationRecord(engine=name, cv_score=score)
                self.comparator.history.add(rec)
            except Exception as exc:
                self.logger.exception("history update failed for %s: %s", name, exc)
                if self.watchdog:
                    self.watchdog.escalate(f"history update failed for {name}: {exc}")
                if RAISE_ERRORS:
                    raise

    def run_cycle(self) -> None:
        try:
            if self.event_bus:
                engines = list(self.manager.engines.keys())
            else:
                engines = [n for n, e in self.manager.engines.items() if e is not None]
            self._results = {}
            if self.event_bus and engines:
                for name in engines:
                    if not publish_with_retry(
                        self.event_bus, "evaluation:run", {"engine": name}
                    ):
                        self.logger.error(
                            "publish failed for engine %s", name
                        )
                        if self.watchdog:
                            self.watchdog.escalate(
                                f"evaluation publish failed for {name}"
                            )
                        if RAISE_ERRORS:
                            raise RuntimeError("publish failed")
                start = time.time()
                while time.time() - start < 5.0 and len(self._results) < len(engines):
                    time.sleep(0.01)
                results = self._results
            else:
                results = self.manager.evaluate_all()
            if results:
                self._persist_results(results)
                self.comparator.evaluate_and_rollback()
                try:
                    best_score = max(
                        (float(r.get("cv_score", 0.0)) for r in results.values()),
                        default=0.0,
                    )
                except Exception:
                    best_score = 0.0
                try:
                    event_id = self.cloner.history.add(
                        EvolutionEvent(
                            action="model_evaluation",
                            before_metric=0.0,
                            after_metric=best_score,
                            roi=best_score,
                            reason="model evaluation cycle",
                            trigger="evaluation",
                            performance=best_score,
                            parent_event_id=self._last_event_id,
                        )
                    )
                    self._last_event_id = event_id
                except Exception as exc:
                    self.logger.exception("history update failed: %s", exc)
                self.cloner.clone_top_workflows(limit=1)
        except Exception as exc:
            self.logger.exception("evaluation cycle failed: %s", exc)
            if self.watchdog:
                self.watchdog.escalate(f"evaluation cycle failed: {exc}")
            if RAISE_ERRORS:
                raise

    def run_continuous(
        self, interval: float = 86400.0, *, stop_event: Event | None = None
    ) -> None:
        """Start the scheduler and return immediately."""

        if self.scheduler:
            return
        use_async = os.getenv("USE_ASYNC_SCHEDULER")
        if use_async:
            sched = _AsyncScheduler()
            sched.add_job(self.run_cycle, interval, "model_evaluation")
            self.scheduler = sched
        elif BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(
                self.run_cycle,
                "interval",
                seconds=interval,
                id="model_evaluation",
            )
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(self.run_cycle, interval, "model_evaluation")
            self.scheduler = sched
        self._stop = stop_event or Event()

    def stop(self) -> None:
        if not self.scheduler:
            return
        if hasattr(self, "_stop") and self._stop:
            self._stop.set()
        if BackgroundScheduler and isinstance(self.scheduler, BackgroundScheduler):
            self.scheduler.shutdown(wait=False)
        else:
            self.scheduler.shutdown()
        self.scheduler = None


__all__ = ["ModelEvaluationService"]
