from __future__ import annotations

from __future__ import annotations

"""Scheduler running self-coding when metrics degrade."""

import threading
import time
import logging
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

import yaml
from dynamic_path_router import resolve_path, path_for_prompt

from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner

try:  # pragma: no cover - optional self-coding dependency
    from .self_coding_manager import SelfCodingManager
except ImportError:  # pragma: no cover - self-coding unavailable
    SelfCodingManager = Any  # type: ignore
from .data_bot import DataBot
from .advanced_error_management import AutomatedRollbackManager
from .sandbox_settings import SandboxSettings
from .error_parser import ErrorParser

try:  # pragma: no cover - optional dependency
    from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler
except Exception:  # pragma: no cover - scheduler utilities may be missing
    _SimpleScheduler = None  # type: ignore[misc]
    BackgroundScheduler = None  # type: ignore[misc]


class SelfCodingScheduler:
    """Trigger :class:`SelfCodingManager` at regular intervals.

    Defaults for ``interval`` are sourced from :class:`SandboxSettings`.
    """

    def __init__(
        self,
        manager: SelfCodingManager,
        data_bot: DataBot,
        *,
        rollback_mgr: AutomatedRollbackManager | None = None,
        nodes: Optional[Iterable[str]] = None,
        interval: int | None = None,
        patch_path: Path | None = None,
        description: str = "auto_patch",
        settings: SandboxSettings | None = None,
        scan_interval: float | None = None,
    ) -> None:
        self.manager = manager
        self.data_bot = data_bot
        self.rollback_mgr = rollback_mgr
        self.nodes: List[str] = list(nodes or [])
        self.settings = settings or SandboxSettings()
        self.interval = interval if interval is not None else self.settings.self_coding_interval
        self.patch_path = Path(patch_path) if patch_path else resolve_path("auto_helpers.py")
        self.description = description
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._thread: Optional[threading.Thread] = None
        self.scan_interval = scan_interval
        self._scan_scheduler: object | None = None

    def _latest_patch_id(self) -> str | None:
        patch_db = getattr(self.manager.engine, "patch_db", None)
        if not patch_db:
            return None
        try:
            row = patch_db.conn.execute(
                "SELECT id FROM patch_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except Exception:
            return None
        return str(row[0]) if row else None

    # ------------------------------------------------------------------
    def _record_cycle_metrics(self, success: bool, retries: int) -> None:
        """Persist outcome of a self-coding cycle to ``sandbox_metrics.yaml``."""

        path = resolve_path("sandbox_metrics.yaml")
        data: Dict[str, Dict[str, float]] = {}
        try:
            if path.exists():
                data = yaml.safe_load(path.read_text()) or {}
        except Exception:  # pragma: no cover - best effort
            self.logger.warning(
                "failed to load sandbox metrics from %s",
                path_for_prompt(path),
                exc_info=True,
            )
            data = {}

        extra = data.setdefault("extra_metrics", {})
        extra["self_coding_cycle_success"] = 1.0 if success else 0.0
        extra["self_coding_cycle_retries"] = float(retries)

        try:
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        except Exception:  # pragma: no cover - best effort
            self.logger.warning(
                "failed to record sandbox metrics at %s",
                path_for_prompt(path),
                exc_info=True,
            )

    # ------------------------------------------------------------------
    def _scan_job(self) -> None:
        start = time.perf_counter()
        suggestions: List[object] = []
        success = False
        try:
            engine = getattr(self.manager, "engine", None)
            if engine:
                suggestions = list(engine.scan_repo())
                success = True
        except Exception:
            self.logger.exception("repo scan failed")
        duration = time.perf_counter() - start
        event_bus = getattr(getattr(self.manager, "engine", None), "event_bus", None)
        if event_bus:
            try:
                event_bus.publish(
                    "self_coding:scan",
                    {
                        "duration": duration,
                        "suggestions": len(suggestions),
                        "success": success,
                    },
                )
            except Exception:
                self.logger.exception("failed to publish scan metrics")

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            try:
                before = self.data_bot.roi(self.manager.bot_name)
                runner = WorkflowSandboxRunner()
                attempt = 0
                success = False
                while attempt < 3 and not success:
                    attempt += 1

                    def _run_patch() -> None:
                        outcome = self.manager.auto_run_patch(
                            self.patch_path,
                            self.description,
                        )
                        summary = outcome.get("summary") if outcome else None
                        failed_tests = int(summary.get("self_tests", {}).get("failed", 0)) if summary else 0
                        if summary is None or failed_tests:
                            patch_id = outcome.get("patch_id") if outcome else None
                            engine = getattr(self.manager, "engine", None)
                            if patch_id is not None and hasattr(engine, "rollback_patch"):
                                try:
                                    engine.rollback_patch(str(patch_id))
                                except Exception:
                                    self.logger.exception("scheduler rollback failed")
                            if summary is None:
                                raise RuntimeError("post validation summary unavailable")
                            raise RuntimeError(f"self tests failed ({failed_tests})")

                    metrics = runner.run(_run_patch, safe_mode=True)
                    module = metrics.modules[0] if getattr(metrics, "modules", None) else None
                    success = bool(
                        getattr(module, "success", getattr(module, "result", False))
                    )
                    if success:
                        break
                    trace = getattr(module, "exception", "") or ""
                    if trace:
                        failure = ErrorParser.parse_failure(str(trace))
                        exc = failure.get("strategy_tag", "")
                        if exc in {"syntax_error", "import_error"}:
                            break

                after = self.data_bot.roi(self.manager.bot_name)
                if after < before:
                    pid = self._latest_patch_id()
                    if pid:
                        self.manager.engine.rollback_patch(pid)
                        if self.rollback_mgr and self.nodes:
                            try:
                                self.rollback_mgr.auto_rollback(pid, self.nodes)
                            except Exception:
                                self.logger.exception("auto rollback failed")
                self._record_cycle_metrics(success, attempt - 1)
            except Exception:
                self.logger.exception("self-coding loop failed")
            time.sleep(self.interval)

    def _scan_loop(self) -> None:
        while self.running and self.scan_interval:
            time.sleep(self.scan_interval)
            self._scan_job()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.scan_interval:
            try:
                if BackgroundScheduler:
                    sched = BackgroundScheduler()
                    sched.add_job(
                        self._scan_job,
                        "interval",
                        seconds=self.scan_interval,
                        id="self_coding_repo_scan",
                    )
                    sched.start()
                    self._scan_scheduler = sched
                elif _SimpleScheduler:
                    sched = _SimpleScheduler()
                    sched.add_job(self._scan_job, self.scan_interval, "self_coding_repo_scan")
                    self._scan_scheduler = sched
                else:
                    t = threading.Thread(target=self._scan_loop, daemon=True)
                    t.start()
                    self._scan_scheduler = t
            except Exception:
                self.logger.exception("failed to schedule repo scans")

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None
        sched = self._scan_scheduler
        if sched:
            try:
                if BackgroundScheduler and isinstance(sched, BackgroundScheduler):
                    sched.shutdown(wait=False)
                elif hasattr(sched, "shutdown"):
                    sched.shutdown()
                elif isinstance(sched, threading.Thread):
                    sched.join(timeout=0)
            except Exception:
                self.logger.exception("failed to stop repo scan scheduler")
            self._scan_scheduler = None


__all__ = ["SelfCodingScheduler"]
