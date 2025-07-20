from __future__ import annotations

"""Service running self tests on a schedule."""

import logging
import os
import shlex
import json
import tempfile
import asyncio
import sys
from typing import Callable, Any
from .error_bot import ErrorDB
from .error_logger import ErrorLogger
from .data_bot import DataBot


class SelfTestService:
    """Periodically execute the test suite to validate core bots."""

    def __init__(
        self,
        db: ErrorDB | None = None,
        *,
        pytest_args: str | None = None,
        workers: int | None = None,
        data_bot: DataBot | None = None,
        result_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_logger = ErrorLogger(db)
        self.data_bot = data_bot
        self.result_callback = result_callback
        self.results: dict[str, Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._async_stop: asyncio.Event | None = None
        env_args = os.getenv("SELF_TEST_ARGS") if pytest_args is None else pytest_args
        self.pytest_args = shlex.split(env_args) if env_args else []
        env_workers = os.getenv("SELF_TEST_WORKERS") if workers is None else workers
        try:
            self.workers = int(env_workers) if env_workers is not None else 1
        except ValueError:
            self.workers = 1

    # ------------------------------------------------------------------
    async def _run_once(self) -> None:
        other_args = [a for a in self.pytest_args if a.startswith("-")]
        paths = [a for a in self.pytest_args if not a.startswith("-")]
        if not paths:
            paths = [None]

        procs: list[tuple[asyncio.subprocess.Process, str]] = []
        passed = 0
        failed = 0
        coverage_total = 0.0
        runtime_total = 0.0

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
            if self.workers > 1:
                cmd.extend(["-n", str(self.workers)])
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
                cov_info = report.get("coverage", {}) or report.get("cov", {})
                coverage_total += float(
                    cov_info.get("percent")
                    or cov_info.get("coverage")
                    or cov_info.get("percent_covered")
                    or 0.0
                )
                runtime_total += float(
                    report.get("duration")
                    or summary.get("duration")
                    or summary.get("runtime")
                    or 0.0
                )
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

        coverage = coverage_total / max(len(procs), 1)
        runtime = runtime_total
        self.results = {
            "passed": passed,
            "failed": failed,
            "coverage": coverage,
            "runtime": runtime,
        }

        if self.result_callback:
            try:
                self.result_callback(self.results)
            except Exception:
                self.logger.exception("result callback failed")

        if self.data_bot:
            try:
                self.data_bot.db.log_eval("self_tests", "coverage", float(coverage))
                self.data_bot.db.log_eval("self_tests", "runtime", float(runtime))
            except Exception:
                self.logger.exception("failed to store metrics")

    async def _schedule_loop(self, interval: float) -> None:
        assert self._async_stop is not None
        while not self._async_stop.is_set():
            try:
                await self._run_once()
            except Exception:
                self.logger.exception("self test run failed")
            try:
                await asyncio.wait_for(self._async_stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    def run_continuous(
        self,
        interval: float = 86400.0,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> asyncio.Task:
        """Start the background schedule loop on *loop*."""

        if self._task:
            return self._task
        self._loop = loop or asyncio.get_event_loop()
        self._async_stop = asyncio.Event()
        self._task = self._loop.create_task(self._schedule_loop(interval))
        return self._task

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        """Stop the schedule loop and wait for completion."""

        if not self._task:
            return
        assert self._async_stop is not None
        self._async_stop.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None


__all__ = ["SelfTestService"]
