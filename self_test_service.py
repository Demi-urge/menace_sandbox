from __future__ import annotations

"""Service running self tests on a schedule."""

import asyncio
import json
import logging
import os
import shlex
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .data_bot import DataBot
from .error_bot import ErrorDB
from .error_logger import ErrorLogger


class SelfTestService:
    """Periodically execute the test suite to validate core bots.

    If ``result_callback`` is provided, it will be invoked with a dictionary
    containing cumulative results each time a test file finishes running and
    again once the entire run completes.  This allows callers to display
    incremental progress while the tests execute.
    """

    def __init__(
        self,
        db: ErrorDB | None = None,
        *,
        pytest_args: str | None = None,
        workers: int | None = None,
        data_bot: DataBot | None = None,
        result_callback: Callable[[dict[str, Any]], Any] | None = None,
        container_image: str = "python:3.11-slim",
        use_container: bool = False,
        history_path: str | Path | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_logger = ErrorLogger(db)
        self.data_bot = data_bot
        self.result_callback = result_callback
        self.container_image = container_image
        self.use_container = use_container
        self.results: dict[str, Any] | None = None
        self.history_path = Path(history_path) if history_path else None
        self._history_db: sqlite3.Connection | None = None
        if self.history_path and self.history_path.suffix == ".db":
            self._history_db = sqlite3.connect(self.history_path)
            self._history_db.execute(
                """
                CREATE TABLE IF NOT EXISTS test_history(
                    passed INTEGER,
                    failed INTEGER,
                    coverage REAL,
                    runtime REAL,
                    ts TEXT
                )
                """
            )
            self._history_db.commit()
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

    def _store_history(self, rec: dict[str, Any]) -> None:
        if not self.history_path:
            return
        try:
            if self._history_db:
                self._history_db.execute(
                    "INSERT INTO test_history(passed, failed, coverage, runtime, ts) VALUES(?,?,?,?,?)",
                    (
                        int(rec["passed"]),
                        int(rec["failed"]),
                        float(rec["coverage"]),
                        float(rec["runtime"]),
                        rec["ts"],
                    ),
                )
                self._history_db.commit()
            else:
                data = []
                if self.history_path.exists():
                    with open(self.history_path, "r", encoding="utf-8") as fh:
                        try:
                            data = json.load(fh) or []
                        except Exception:
                            data = []
                data.append(rec)
                with open(self.history_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
        except Exception:
            self.logger.exception("failed to store history")

    async def _docker_available(self) -> bool:
        """Return ``True`` if the docker CLI is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            return False
        except Exception:
            self.logger.exception("docker check failed")
            return False

    # ------------------------------------------------------------------
    async def _run_once(self) -> None:
        other_args = [a for a in self.pytest_args if a.startswith("-")]
        paths = [a for a in self.pytest_args if not a.startswith("-")]
        if not paths:
            paths = [None]

        passed = 0
        failed = 0
        coverage_total = 0.0
        runtime_total = 0.0
        proc_info: list[tuple[asyncio.subprocess.Process, str | None]] = []

        use_container = self.use_container and await self._docker_available()
        use_pipe = self.result_callback is not None or use_container

        for p in paths:
            tmp_name: str | None = None
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--json-report",
            ]
            if use_pipe:
                cmd.append("--json-report-file=-")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                tmp.close()
                tmp_name = tmp.name
                cmd.append(f"--json-report-file={tmp_name}")
            cmd.extend(other_args)
            if self.workers > 1:
                cmd.extend(["-n", str(self.workers)])
            if p:
                cmd.append(p)

            if use_container:
                docker_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    "-v",
                    f"{os.getcwd()}:{os.getcwd()}",
                    "-w",
                    os.getcwd(),
                ]
                for k, v in os.environ.items():
                    docker_cmd.extend(["-e", f"{k}={v}"])
                docker_cmd.append(self.container_image)
                docker_cmd.extend(cmd)
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=(
                        asyncio.subprocess.PIPE
                        if use_pipe
                        else asyncio.subprocess.DEVNULL
                    ),
                    stderr=asyncio.subprocess.DEVNULL,
                )
                proc_info.append((proc, tmp_name))
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=(
                        asyncio.subprocess.PIPE
                        if use_pipe
                        else asyncio.subprocess.DEVNULL
                    ),
                    stderr=asyncio.subprocess.DEVNULL,
                )
                proc_info.append((proc, tmp_name))

        async def _process(proc: asyncio.subprocess.Process, tmp: str | None) -> tuple[int, int, float, float]:
            await proc.wait()
            report: dict[str, Any] = {}
            try:
                if use_pipe:
                    assert proc.stdout is not None
                    out = await proc.stdout.read()
                    report = json.loads(out.decode() or "{}")
                else:
                    assert tmp is not None
                    with open(tmp, "r", encoding="utf-8") as fh:
                        report = json.load(fh)
            except Exception:
                self.logger.exception("failed to parse test report")
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass

            summary = report.get("summary", {})
            pcount = int(summary.get("passed", 0))
            fcount = int(summary.get("failed", 0)) + int(summary.get("error", 0))
            cov_info = report.get("coverage", {}) or report.get("cov", {})
            cov = float(
                cov_info.get("percent")
                or cov_info.get("coverage")
                or cov_info.get("percent_covered")
                or 0.0
            )
            runtime = float(
                report.get("duration")
                or summary.get("duration")
                or summary.get("runtime")
                or 0.0
            )

            if proc.returncode != 0:
                exc = RuntimeError(
                    f"self tests failed with code {proc.returncode}"
                )
                self.logger.error("self tests failed: %s", exc)
                try:
                    self.error_logger.log(exc, "self_tests", "sandbox")
                except Exception:
                    self.logger.exception("error logging failed")
                raise exc

            return pcount, fcount, cov, runtime

        tasks = [asyncio.create_task(_process(p, t)) for p, t in proc_info]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        first_exc: Exception | None = None

        for res in results:
            if isinstance(res, Exception):
                if first_exc is None:
                    first_exc = res
                continue
            pcount, fcount, cov, runtime = res
            passed += pcount
            failed += fcount
            coverage_total += cov
            runtime_total += runtime

            if self.result_callback:
                partial = {
                    "passed": passed,
                    "failed": failed,
                    "coverage": coverage_total / max(len(paths), 1),
                    "runtime": runtime_total,
                }
                try:
                    self.result_callback(partial)
                except Exception:
                    self.logger.exception("result callback failed")

        try:
            self.error_logger.db.add_test_result(passed, failed)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to store test results")

        coverage = coverage_total / max(len(proc_info), 1)
        runtime = runtime_total
        self.results = {
            "passed": passed,
            "failed": failed,
            "coverage": coverage,
            "runtime": runtime,
        }
        self._store_history(
            {
                "passed": passed,
                "failed": failed,
                "coverage": coverage,
                "runtime": runtime,
                "ts": datetime.utcnow().isoformat(),
            }
        )

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

        if first_exc:
            raise first_exc

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
    def recent_history(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.history_path:
            return []
        try:
            if self._history_db:
                cur = self._history_db.execute(
                    "SELECT passed, failed, coverage, runtime, ts FROM test_history ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "passed": int(r[0]),
                        "failed": int(r[1]),
                        "coverage": float(r[2]),
                        "runtime": float(r[3]),
                        "ts": r[4],
                    }
                    for r in rows
                ]
            else:
                if not self.history_path.exists():
                    return []
                with open(self.history_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or []
                return list(reversed(data))[:limit]
        except Exception:
            self.logger.exception("failed to read history")
            return []

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


__all__ = ["SelfTestService", "cli", "main"]


def cli(argv: list[str] | None = None) -> int:
    """Command line interface for running the self tests."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run self tests once")
    run.add_argument("paths", nargs="*", help="Test paths or patterns")
    run.add_argument("--workers", type=int, default=1, help="Number of pytest workers")
    run.add_argument(
        "--container-image",
        default="python:3.11-slim",
        help="Docker image when using containers",
    )
    run.add_argument(
        "--use-container",
        action="store_true",
        help="Execute tests inside a Docker container",
    )
    run.add_argument(
        "--history",
        help="Path to JSON/DB file storing run history",
    )
    run.add_argument(
        "--pytest-args",
        default=None,
        help="Additional arguments passed to pytest",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        pytest_args = []
        if args.pytest_args:
            pytest_args.extend(shlex.split(args.pytest_args))
        if args.paths:
            pytest_args.extend(args.paths)
        service = SelfTestService(
            pytest_args=" ".join(pytest_args) if pytest_args else None,
            workers=args.workers,
            container_image=args.container_image,
            use_container=args.use_container,
            history_path=args.history,
        )
        try:
            asyncio.run(service._run_once())
        except Exception as exc:
            print(f"self test run failed: {exc}", file=sys.stderr)
            return 1
        return 0

    parser.error("unknown command")
    return 1


def main(argv: list[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
