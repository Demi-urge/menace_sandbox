from __future__ import annotations

"""Restart sandbox runs when unexpected failures occur."""

from typing import Any, Callable, Dict
import argparse
import logging
import os
from pathlib import Path
import time
import traceback

try:
    from .metrics_exporter import CollectorRegistry
except Exception:  # pragma: no cover - module may not be a package
    from metrics_exporter import CollectorRegistry  # type: ignore

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.sandbox_main = sandbox_main
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)
        self.on_retry = on_retry
        self.restart_count = 0
        self.last_failure_time: float | None = None

        self._restart_gauge = None
        self._failure_gauge = None
        try:
            from .metrics_exporter import Gauge

            self._restart_gauge = Gauge(
                "sandbox_restart_count",
                "Number of sandbox restarts",
                registry=registry,
            )
            self._failure_gauge = Gauge(
                "sandbox_last_failure_time",
                "Timestamp of last sandbox failure",
                registry=registry,
            )
        except Exception:  # pragma: no cover - optional dependency missing
            pass

    # ------------------------------------------------------------------
    @property
    def metrics(self) -> Dict[str, float | None]:
        """Return restart count and last failure time."""
        return {
            "restart_count": float(self.restart_count),
            "last_failure_time": self.last_failure_time,
        }

    # ------------------------------------------------------------------
    def run(self, preset: Dict[str, Any], args: argparse.Namespace):
        """Execute ``sandbox_main`` retrying on failure."""
        attempts = 0
        while True:
            start = time.monotonic()
            try:
                return self.sandbox_main(preset, args)
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
                        f"{ts} attempt={attempts} runtime={runtime:.2f}s\n{tb}\n"
                    )

                if self.on_retry:
                    try:
                        self.on_retry(exc, runtime)
                    except Exception:
                        self.logger.exception("on_retry callback failed")

                if self._restart_gauge:
                    try:
                        self._restart_gauge.set(float(self.restart_count))
                        if self.last_failure_time is not None:
                            self._failure_gauge.set(self.last_failure_time)
                    except Exception:  # pragma: no cover - runtime issues
                        self.logger.exception("failed to update metrics")

                if self.max_retries is not None and attempts >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)


__all__ = ["SandboxRecoveryManager"]
