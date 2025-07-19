from __future__ import annotations

"""Service running self tests on a schedule."""

import logging
import os
import shlex
import json
import tempfile
import asyncio
import sys
from threading import Event

from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler, _AsyncScheduler
from .error_bot import ErrorDB
from .error_logger import ErrorLogger


class SelfTestService:
    """Periodically execute the test suite to validate core bots."""

    def __init__(self, db: ErrorDB | None = None, *, pytest_args: str | None = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_logger = ErrorLogger(db)
        self.scheduler: object | None = None
        env_args = os.getenv("SELF_TEST_ARGS") if pytest_args is None else pytest_args
        self.pytest_args = shlex.split(env_args) if env_args else []

    # ------------------------------------------------------------------
    async def _run_once(self) -> None:
        other_args = [a for a in self.pytest_args if a.startswith("-")]
        paths = [a for a in self.pytest_args if not a.startswith("-")]
        if not paths:
            paths = [None]

        procs: list[tuple[asyncio.subprocess.Process, str]] = []
        passed = 0
        failed = 0

        for p in paths:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp.close()
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--json-report",
                f"--json-report-file={tmp.name}",
                *other_args,
            ]
            if p:
                cmd.append(p)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            procs.append((proc, tmp.name))

        await asyncio.gather(*(pr.wait() for pr, _ in procs))

        for proc, tmp_name in procs:
            try:
                with open(tmp_name, "r", encoding="utf-8") as fh:
                    report = json.load(fh)
                summary = report.get("summary", {})
                passed += int(summary.get("passed", 0))
                failed += int(summary.get("failed", 0)) + int(summary.get("error", 0))
            except Exception:
                self.logger.exception("failed to parse test report")
            finally:
                try:
                    os.unlink(tmp_name)
                except Exception:
                    pass
            if proc.returncode != 0:
                exc = RuntimeError(f"self tests failed with code {proc.returncode}")
                self.logger.error("self tests failed: %s", exc)
                try:
                    self.error_logger.log(exc, "self_tests", "sandbox")
                except Exception:
                    self.logger.exception("error logging failed")

        try:
            self.error_logger.db.add_test_result(passed, failed)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to store test results")

    # ------------------------------------------------------------------
    def run_continuous(self, interval: float = 86400.0, *, stop_event: Event | None = None) -> None:
        if self.scheduler:
            return
        use_async = os.getenv("USE_ASYNC_SCHEDULER")
        if use_async:
            sched = _AsyncScheduler()
            sched.add_job(self._run_once, interval, "self_test")
            self.scheduler = sched
        elif BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(lambda: asyncio.run(self._run_once()), "interval", seconds=interval, id="self_test")
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(lambda: asyncio.run(self._run_once()), interval, "self_test")
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


__all__ = ["SelfTestService"]
