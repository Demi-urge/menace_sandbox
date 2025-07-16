from __future__ import annotations

"""Background service for continuous compliance auditing."""

import logging
import os
import json
import re
import signal
from pathlib import Path
from threading import Event
from typing import Optional, Union, Callable, Dict, List

from .compliance_checker import ComplianceChecker
from .security_auditor import SecurityAuditor
from .cross_model_scheduler import _SimpleScheduler
from .retry_utils import with_retry

try:  # pragma: no cover - optional config support
    from .config_loader import get_config_value
except Exception:  # pragma: no cover - config loader optional
    def get_config_value(key_path: str, default: Optional[float] = None) -> Optional[float]:
        return default

try:  # pragma: no cover - optional dependency
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - APScheduler missing
    BackgroundScheduler = None  # type: ignore


DEFAULT_AUDIT_INTERVAL = 3600.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0

LOG_LINE_RE = re.compile(r"^\s*(?P<sig>[A-Za-z0-9+/=]+)\s+(?P<json>\{.*\})\s*$")


def _validate_float(name: str, value: object, default: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("invalid %s value %r", name, value)
        return default
    if val <= 0:
        logging.getLogger(__name__).warning("%s must be positive", name)
        return default
    return val


def _validate_int(name: str, value: object, default: int) -> int:
    try:
        val = int(value)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("invalid %s value %r", name, value)
        return default
    if val <= 0:
        logging.getLogger(__name__).warning("%s must be positive", name)
        return default
    return val


def _get_interval() -> float:
    env = os.getenv("AUDIT_INTERVAL")
    cfg = get_config_value("audit.interval", env if env is not None else DEFAULT_AUDIT_INTERVAL)
    return _validate_float("audit.interval", cfg, DEFAULT_AUDIT_INTERVAL)


def _get_retry_attempts() -> int:
    env = os.getenv("AUDIT_RETRY_ATTEMPTS")
    cfg = get_config_value("audit.retry_attempts", env if env is not None else DEFAULT_RETRY_ATTEMPTS)
    return _validate_int("audit.retry_attempts", cfg, DEFAULT_RETRY_ATTEMPTS)


def _get_retry_delay() -> float:
    env = os.getenv("AUDIT_RETRY_DELAY")
    cfg = get_config_value("audit.retry_delay", env if env is not None else DEFAULT_RETRY_DELAY)
    return _validate_float("audit.retry_delay", cfg, DEFAULT_RETRY_DELAY)


class ComplianceAuditService:
    """Periodically run security audits and inspect compliance logs."""

    def __init__(
        self,
        checker: Optional[ComplianceChecker] = None,
        auditor: Optional[SecurityAuditor] = None,
        *,
        on_violation: Optional[Callable[[Dict[str, int]], None]] = None,
        test_mode: bool = False,
    ) -> None:
        self.checker = checker or ComplianceChecker()
        self.auditor = auditor or SecurityAuditor()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scheduler: Optional[Union[BackgroundScheduler, _SimpleScheduler]] = None
        self.on_violation = on_violation
        self.test_mode = test_mode
        self._orig_signals: Dict[int, Callable] = {}
        self._test_log_lines: List[str] | None = None

    # ------------------------------------------------------------------
    def _parse_log_line(self, line: str) -> Optional[dict]:
        line = line.strip()
        m = LOG_LINE_RE.match(line)
        data = None
        if m:
            try:
                data = json.loads(m.group("json"))
            except json.JSONDecodeError:
                data = None
        if data is None:
            try:
                sig, rest = line.split(None, 1)
                if re.fullmatch(r"[A-Za-z0-9+/=]+", sig):
                    data = json.loads(rest)
            except Exception:
                return None
        return data

    def _run_once(self) -> None:

        def _audit() -> bool:
            ok = self.auditor.audit()
            if not ok:
                raise RuntimeError("audit failed")
            return ok

        try:
            with_retry(
                _audit,
                attempts=_get_retry_attempts(),
                delay=_get_retry_delay(),
                logger=self.logger,
            )
        except Exception as exc:
            self.logger.error("security audit failed after retries: %s", exc)

        log_lines: List[str] | None = self._test_log_lines
        if log_lines is None:
            log_path_attr = getattr(self.checker, "log_path", None)
            if not isinstance(log_path_attr, str):
                self.logger.error("invalid log path: %r", log_path_attr)
                return
            log_path = Path(log_path_attr)
            if log_path.exists() and log_path.stat().st_size > 0:
                try:
                    with log_path.open() as fh:
                        log_lines = list(fh)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.error("failed to read compliance log: %s", exc)
                    log_lines = None
        if log_lines:
            counts: dict[str, int] = {}
            for line in log_lines:
                data = self._parse_log_line(line)
                if not data:
                    self.logger.debug("skipping malformed log line: %s", line.strip())
                    continue
                typ = str(data.get("type", "unknown"))
                counts[typ] = counts.get(typ, 0) + 1
            if counts:
                summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
                self.logger.warning(
                    "compliance violations detected (%s)", summary
                )
                if self.on_violation:
                    try:
                        self.on_violation(counts)
                    except Exception:
                        self.logger.error("violation handler failed", exc_info=True)

    # ------------------------------------------------------------------
    def run_continuous(
        self,
        interval: Optional[float] = None,
        *,
        stop_event: Optional[Event] = None,
        handle_signals: bool = True,
        wait: bool = False,
        scheduler_cls: Optional[type] = None,
    ) -> None:
        if interval is None:
            interval = _get_interval()

        if self.scheduler:
            self.logger.warning("run_continuous called while scheduler is running")
            return

        if scheduler_cls:
            cls = scheduler_cls
        else:
            cls = BackgroundScheduler if BackgroundScheduler else _SimpleScheduler

        if cls is BackgroundScheduler:
            sched = cls()
            sched.add_job(
                self._run_once, "interval", seconds=interval, id="compliance_audit"
            )
            sched.start()
        else:
            sched = cls()
            sched.add_job(self._run_once, interval, "compliance_audit")
        self.scheduler = sched
        self._stop = stop_event or Event()

        if handle_signals and not self.test_mode:
            for sig_no in (signal.SIGINT, signal.SIGTERM):
                self._orig_signals[sig_no] = signal.getsignal(sig_no)
                signal.signal(sig_no, lambda s, f: self.stop())

        if wait:
            try:
                self._stop.wait()
            finally:
                if handle_signals and not self.test_mode:
                    for sig_no, handler in self._orig_signals.items():
                        signal.signal(sig_no, handler)

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
        if self._orig_signals:
            for sig_no, handler in self._orig_signals.items():
                try:
                    signal.signal(sig_no, handler)
                except Exception:
                    pass
            self._orig_signals.clear()


__all__ = ["ComplianceAuditService"]
