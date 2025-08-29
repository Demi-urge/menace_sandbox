from __future__ import annotations

"""Restart sandbox runs when unexpected failures occur."""

from typing import Any, Callable, Dict, List
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

try:
    from .logging_utils import set_correlation_id
except Exception:  # pragma: no cover - module not a package
    from logging_utils import set_correlation_id  # type: ignore

try:
    from .resilience import (
        CircuitBreaker,
        CircuitOpenError,
        ResilienceError,
    )
except Exception:  # pragma: no cover - module not a package
    from resilience import (
        CircuitBreaker,  # type: ignore
        CircuitOpenError,  # type: ignore
        ResilienceError,  # type: ignore
    )

try:
    from .metrics_exporter import CollectorRegistry
except Exception:  # pragma: no cover - module may not be a package
    from metrics_exporter import CollectorRegistry  # type: ignore

logger = logging.getLogger(__name__)


class SandboxRecoveryError(ResilienceError):
    """Raised when sandbox recovery cannot proceed."""


class RecoveryMetricsRecorder:
    """Record metrics via :mod:`metrics_exporter` or a local file."""

    def __init__(self) -> None:
        try:
            try:
                from . import metrics_exporter as _me
            except Exception:  # pragma: no cover - package not available
                import metrics_exporter as _me  # type: ignore

            self._using_exporter = not getattr(_me, "_USING_STUB", False)
            if self._using_exporter:
                self._restart_gauge = _me.sandbox_restart_total
                self._failure_gauge = _me.sandbox_last_failure_ts
            else:
                self._restart_gauge = None
                self._failure_gauge = None
        except Exception:  # pragma: no cover - optional dependency missing
            self._using_exporter = False
            self._restart_gauge = None
            self._failure_gauge = None

    def record(
        self,
        restart_count: int,
        last_failure_ts: float | None,
        data_dir: Path,
    ) -> None:
        ts = float(last_failure_ts) if last_failure_ts is not None else 0.0
        if self._using_exporter and self._restart_gauge and self._failure_gauge:
            try:
                self._restart_gauge.set(float(restart_count))
                self._failure_gauge.set(ts)
            except Exception:  # pragma: no cover - runtime issues
                logger.exception("failed to update metrics")
            return

        payload = {
            "sandbox_restart_total": float(restart_count),
            "sandbox_last_failure_ts": ts,
        }
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(data_dir / "recovery.json", "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception:  # pragma: no cover - runtime issues
            logger.exception("failed to write recovery metrics")


class SandboxRecoveryManager:
    """Wrap ``_sandbox_main`` and restart on uncaught errors."""

    def __init__(
        self,
        sandbox_main: Callable[[Dict[str, Any], argparse.Namespace], Any],
        *,
        retry_delay: float = 1.0,
        max_retries: int | None = None,
        on_retry: Callable[[Exception, float], None] | None = None,
        registry: "CollectorRegistry" | None = None,
        circuit_max_failures: int = 5,
        circuit_reset_timeout: float = 60.0,
    ) -> None:
        self.sandbox_main = sandbox_main
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)
        self.on_retry = on_retry
        self.restart_count = 0
        self.last_failure_time: float | None = None
        self._circuit = CircuitBreaker(
            max_failures=circuit_max_failures, reset_timeout=circuit_reset_timeout
        )

        self._metrics_recorder = RecoveryMetricsRecorder()

    # ------------------------------------------------------------------
    @property
    def metrics(self) -> Dict[str, float | None]:
        """Return restart count and last failure time."""
        return {
            "sandbox_restart_total": float(self.restart_count),
            "sandbox_last_failure_ts": self.last_failure_time,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def load_last_tracker(data_dir: str | Path):
        """Return :class:`ROITracker` loaded from ``data_dir`` or ``None``."""
        try:
            from menace.roi_tracker import ROITracker
        except Exception:  # pragma: no cover - fallback
            from roi_tracker import ROITracker  # type: ignore

        path = Path(data_dir) / "roi_history.json"
        tracker = ROITracker()
        try:
            tracker.load_history(str(path))
        except Exception:
            logger.exception("failed to load tracker history: %s", path)
            return None
        return tracker

    # ------------------------------------------------------------------
    def run(self, preset: Dict[str, Any], args: argparse.Namespace):
        """Execute ``sandbox_main`` retrying on failure."""
        attempts = 0
        delay = self.retry_delay
        while True:
            cid = uuid.uuid4().hex
            set_correlation_id(cid)
            start = time.monotonic()
            try:
                return self._circuit.call(lambda: self.sandbox_main(preset, args))
            except CircuitOpenError as exc:
                self.logger.error("recovery circuit open: %s", exc)
                raise SandboxRecoveryError("circuit open") from exc
            except Exception as exc:  # pragma: no cover - rare
                attempts += 1
                self.restart_count += 1
                self.last_failure_time = time.time()
                runtime = time.monotonic() - start
                self.logger.exception("sandbox run crashed; restarting")

                log_dir = Path(
                    getattr(args, "sandbox_data_dir", None)
                    or os.getenv("SANDBOX_DATA_DIR", "sandbox_data")
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "recovery.log"
                tb = traceback.format_exc()
                with open(log_file, "a", encoding="utf-8") as fh:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    fh.write(
                        f"{ts} cid={cid} attempt={attempts} runtime={runtime:.2f}s\n{tb}\n"
                    )

                if self.on_retry:
                    try:
                        self.on_retry(exc, runtime)
                    except Exception:
                        self.logger.exception("on_retry callback failed")
                self._metrics_recorder.record(
                    self.restart_count, self.last_failure_time, log_dir
                )

                if self.max_retries is not None and attempts >= self.max_retries:
                    raise SandboxRecoveryError("maximum retries reached") from exc
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            finally:
                set_correlation_id(None)


__all__ = [
    "SandboxRecoveryError",
    "SandboxRecoveryManager",
    "cli",
    "load_metrics",
    "load_last_tracker",
]


def load_metrics(path: Path) -> Dict[str, float]:
    """Return metrics stored in ``path`` as ``float`` values."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        logger.exception("failed to load recovery metrics: %s", path)
        return {}
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        mapping = {
            "restart_count": "sandbox_restart_total",
            "last_failure_time": "sandbox_last_failure_ts",
        }
        for k, v in data.items():
            name = mapping.get(str(k), str(k))
            try:
                out[name] = float(v)
            except Exception:
                out[name] = 0.0
    return out


def cli(argv: List[str] | None = None) -> int:
    """Print sandbox recovery metrics."""
    parser = argparse.ArgumentParser(description=cli.__doc__)
    parser.add_argument(
        "--file",
        default=str(Path("sandbox_data") / "recovery.json"),
        help="Path to recovery.json",
    )
    args = parser.parse_args(argv)

    try:
        data = load_metrics(Path(args.file))
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.error("failed to read %s: %s", args.file, exc)
        return 1

    for k, v in data.items():
        logger.info("%s: %s", k, v)
    return 0


def main(argv: List[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
