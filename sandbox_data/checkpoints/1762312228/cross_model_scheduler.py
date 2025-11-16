from __future__ import annotations

"""Schedule periodic model ranking and deployment."""

import json
import logging
import time
from importlib import import_module
from pathlib import Path
from threading import Event
import os
import asyncio

from .cross_model_comparator import CrossModelComparator
from .neuroplasticity import PathwayDB
from .evaluation_history_db import EvaluationHistoryDB
import threading


class _SimpleScheduler:
    """Very small in-process scheduler used when APScheduler is unavailable.

    Jobs are persisted to disk so they can be restored on restart."""

    STATE_FILE = Path("scheduler_state.json")
    RETRY_DELAY = 5.0
    MAX_RETRIES = 3
    MISFIRE_GRACE_TIME = 60.0

    def __init__(self) -> None:
        self.tasks: list[tuple[float, callable, str]] = []
        self._next_runs: dict[str, float] = {}
        self._retry_counts: dict[str, int] = {}
        self._retry_delays: dict[str, float] = {}
        self._max_retries: dict[str, int] = {}
        self._misfire_grace: dict[str, float] = {}
        self.stop = Event()
        self.thread: threading.Thread | None = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.STATE_FILE.exists():
            return
        try:
            data = json.loads(self.STATE_FILE.read_text())
        except Exception:
            self.logger.exception("failed to load scheduler state")
            return
        for job in data.get("jobs", []):
            try:
                module_name, qualname = job["func"].split(":", 1)
                mod = import_module(module_name)
                func = mod
                for part in qualname.split("."):
                    func = getattr(func, part)
                interval = job["interval"]
                jid = job["id"]
                next_run = job.get("next_run", time.time() + interval)
                retries = job.get("retries", 0)
                retry_delay = job.get("retry_delay", self.RETRY_DELAY)
                max_retries = job.get("max_retries", self.MAX_RETRIES)
                misfire_grace = job.get("misfire_grace", self.MISFIRE_GRACE_TIME)
                self.tasks.append((interval, func, jid))
                self._next_runs[jid] = next_run
                self._retry_counts[jid] = retries
                self._retry_delays[jid] = retry_delay
                self._max_retries[jid] = max_retries
                self._misfire_grace[jid] = misfire_grace
            except Exception:
                self.logger.exception("failed to restore job %s", job)
        if self.tasks and not self.thread:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        jobs = []
        for interval, func, jid in self.tasks:
            func_path = f"{func.__module__}:{func.__qualname__}"
            jobs.append(
                {
                    "id": jid,
                    "interval": interval,
                    "func": func_path,
                    "next_run": self._next_runs.get(jid, time.time() + interval),
                    "retries": self._retry_counts.get(jid, 0),
                    "retry_delay": self._retry_delays.get(jid, self.RETRY_DELAY),
                    "max_retries": self._max_retries.get(jid, self.MAX_RETRIES),
                    "misfire_grace": self._misfire_grace.get(jid, self.MISFIRE_GRACE_TIME),
                }
            )
        try:
            self.STATE_FILE.write_text(json.dumps({"jobs": jobs}))
        except Exception:
            self.logger.exception("failed to save scheduler state")

    def add_job(
        self,
        func: callable,
        interval: float,
        id: str,
        *,
        retry_delay: float | None = None,
        max_retries: int | None = None,
        misfire_grace_time: float | None = None,
    ) -> None:
        self.tasks.append((interval, func, id))
        self._next_runs[id] = time.time() + interval
        self._retry_counts[id] = 0
        self._retry_delays[id] = (
            self.RETRY_DELAY if retry_delay is None else retry_delay
        )
        self._max_retries[id] = (
            self.MAX_RETRIES if max_retries is None else max_retries
        )
        self._misfire_grace[id] = (
            self.MISFIRE_GRACE_TIME if misfire_grace_time is None else misfire_grace_time
        )
        self._save_state()
        if not self.thread:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def list_jobs(self) -> list[str]:
        """Return identifiers of scheduled jobs."""
        return [jid for _, _, jid in self.tasks]

    def remove_job(self, id: str) -> None:
        """Remove a scheduled job by id."""
        self.tasks = [t for t in self.tasks if t[2] != id]
        self._next_runs.pop(id, None)
        self._retry_counts.pop(id, None)
        self._retry_delays.pop(id, None)
        self._max_retries.pop(id, None)
        self._misfire_grace.pop(id, None)
        self._save_state()

    def _run(self) -> None:
        while not self.stop.is_set():
            now = time.time()
            sleeps: list[float] = []
            for interval, fn, jid in list(self.tasks):
                next_run = self._next_runs.get(jid, now)
                if now >= next_run:
                    misfire_grace = self._misfire_grace.get(jid, self.MISFIRE_GRACE_TIME)
                    if now - next_run > misfire_grace:
                        self.logger.warning("job %s misfired by %.1fs", jid, now - next_run)
                        self._next_runs[jid] = now + interval
                        sleeps.append(self._next_runs[jid] - now)
                        continue
                    try:
                        fn()
                        self._retry_counts[jid] = 0
                        self._next_runs[jid] = now + interval
                    except BaseException:
                        if not self.stop.is_set():
                            self.logger.exception("job %s failed", jid)
                        else:
                            raise
                        count = self._retry_counts.get(jid, 0) + 1
                        self._retry_counts[jid] = count
                        delay = self._retry_delays.get(jid, self.RETRY_DELAY)
                        max_tries = self._max_retries.get(jid, self.MAX_RETRIES)
                        if count <= max_tries:
                            self._next_runs[jid] = now + delay * count
                        else:
                            self._next_runs[jid] = now + interval
                    sleeps.append(self._next_runs[jid] - now)
            time.sleep(max(0.0, min(sleeps, default=0.1)))
            self._save_state()

    def shutdown(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=0)
        self._save_state()


class _AsyncScheduler:
    """Asynchronous in-process scheduler."""

    RETRY_DELAY = 5.0
    MAX_RETRIES = 3
    MISFIRE_GRACE_TIME = 60.0

    def __init__(self) -> None:
        self.tasks: list[tuple[float, callable, str]] = []
        self._next_runs: dict[str, float] = {}
        self._retry_counts: dict[str, int] = {}
        self._retry_delays: dict[str, float] = {}
        self._max_retries: dict[str, int] = {}
        self._misfire_grace: dict[str, float] = {}
        self.stop = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def add_job(
        self,
        func: callable,
        interval: float,
        id: str,
        *,
        retry_delay: float | None = None,
        max_retries: int | None = None,
        misfire_grace_time: float | None = None,
    ) -> None:
        self.tasks.append((interval, func, id))
        self._next_runs[id] = time.time() + interval
        self._retry_counts[id] = 0
        self._retry_delays[id] = (
            self.RETRY_DELAY if retry_delay is None else retry_delay
        )
        self._max_retries[id] = (
            self.MAX_RETRIES if max_retries is None else max_retries
        )
        self._misfire_grace[id] = (
            self.MISFIRE_GRACE_TIME if misfire_grace_time is None else misfire_grace_time
        )
        if not self.thread:
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def list_jobs(self) -> list[str]:
        return [jid for _, _, jid in self.tasks]

    def remove_job(self, id: str) -> None:
        self.tasks = [t for t in self.tasks if t[2] != id]
        self._next_runs.pop(id, None)
        self._retry_counts.pop(id, None)
        self._retry_delays.pop(id, None)
        self._max_retries.pop(id, None)
        self._misfire_grace.pop(id, None)

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._run())
        self.loop.run_forever()

    async def _run(self) -> None:
        while not self.stop.is_set():
            now = time.time()
            sleeps: list[float] = []
            for interval, fn, jid in list(self.tasks):
                next_run = self._next_runs.get(jid, now)
                if now >= next_run:
                    misfire_grace = self._misfire_grace.get(jid, self.MISFIRE_GRACE_TIME)
                    if now - next_run > misfire_grace:
                        self.logger.warning("job %s misfired by %.1fs", jid, now - next_run)
                        self._next_runs[jid] = time.time() + interval
                        sleeps.append(self._next_runs[jid] - time.time())
                        continue
                    try:
                        result = fn()
                        if asyncio.iscoroutine(result):
                            await result
                        self._retry_counts[jid] = 0
                        self._next_runs[jid] = time.time() + interval
                    except BaseException:
                        if not self.stop.is_set():
                            self.logger.exception("job %s failed", jid)
                        else:
                            raise
                        count = self._retry_counts.get(jid, 0) + 1
                        self._retry_counts[jid] = count
                        delay = self._retry_delays.get(jid, self.RETRY_DELAY)
                        max_tries = self._max_retries.get(jid, self.MAX_RETRIES)
                        if count <= max_tries:
                            self._next_runs[jid] = time.time() + delay * count
                        else:
                            self._next_runs[jid] = time.time() + interval
                    sleeps.append(self._next_runs[jid] - time.time())
            await asyncio.sleep(max(0.0, min(sleeps, default=0.1)))

    def shutdown(self) -> None:
        self.stop.set()
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=0)


try:  # pragma: no cover - optional dependency
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - APScheduler missing
    BackgroundScheduler = None  # type: ignore


class ModelRankingService:
    """Periodically call :meth:`CrossModelComparator.rank_and_deploy`."""

    def __init__(self, comparator: CrossModelComparator | None = None) -> None:
        self.comparator = comparator or CrossModelComparator(
            PathwayDB(), EvaluationHistoryDB()
        )
        self.logger = logging.getLogger("ModelRankingService")
        self.scheduler: object | None = None

    # ------------------------------------------------------------------
    def run_continuous(
        self,
        interval: float = 86400.0,
        *,
        stop_event: Event | None = None,
    ) -> None:
        """Start the scheduler and return immediately."""

        if self.scheduler:
            return
        use_async = os.getenv("USE_ASYNC_SCHEDULER")
        if use_async:
            sched = _AsyncScheduler()
            sched.add_job(self.comparator.rank_and_deploy, interval, "model_ranking")
            self.scheduler = sched
        elif BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(
                self.comparator.rank_and_deploy,
                "interval",
                seconds=interval,
                id="model_ranking",
            )
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(self.comparator.rank_and_deploy, interval, "model_ranking")
            self.scheduler = sched
        self._stop = stop_event or Event()

    # ------------------------------------------------------------------
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


__all__ = ["ModelRankingService", "_SimpleScheduler", "_AsyncScheduler"]
