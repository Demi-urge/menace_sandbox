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
        self._using_stub = False
        try:
            try:
                from . import metrics_exporter as _me
            except Exception:  # pragma: no cover - package not available
                import metrics_exporter as _me  # type: ignore

            self._using_stub = getattr(_me, "_USING_STUB", False)
            Gauge = _me.Gauge

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

                if self._using_stub:
                    try:
                        metrics_file = log_dir / "recovery.json"
                        with open(metrics_file, "w", encoding="utf-8") as fh:
                            json.dump(self.metrics, fh)
                    except Exception:  # pragma: no cover - runtime issues
                        self.logger.exception("failed to write recovery metrics")

                if self.max_retries is not None and attempts >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)


__all__ = ["SandboxRecoveryManager", "cli"]


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
        with open(args.file, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - runtime issues
        print(f"failed to read {args.file}: {exc}", file=sys.stderr)
        return 1

    for k, v in data.items():
        print(f"{k}: {v}")
    return 0


def main(argv: List[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
