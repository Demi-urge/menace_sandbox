import json
import logging
import os
import threading
from pathlib import Path
import time
import asyncio
from filelock import FileLock

try:  # pragma: no cover - prefer package import
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - allow running as script
    from dynamic_path_router import resolve_path  # type: ignore

if os.getenv("SANDBOX_CENTRAL_LOGGING") == "1":
    from logging_utils import setup_logging

    setup_logging()

# ``metrics_exporter`` is optional and in tests may be loaded via a custom
# ``importlib`` loader that leaves ``sys.modules['menace.metrics_exporter']`` as
# ``None``.  Importing it again would reset any gauges already configured, so we
# first try to reuse an existing module instance before falling back to a real
# import.
try:
    import sys
    _pkg = sys.modules.get(__package__ or "")
    _me = getattr(_pkg, "metrics_exporter", None) if _pkg else None
    if _me is None:  # pragma: no cover - executed only when not preloaded
        from . import metrics_exporter as _me  # type: ignore

    Gauge = _me.Gauge  # type: ignore
    start_metrics_server = _me.start_metrics_server  # type: ignore
    synergy_weight_update_alerts_total = _me.synergy_weight_update_alerts_total  # type: ignore
    synergy_weight_update_failures_total = _me.synergy_weight_update_failures_total  # type: ignore
except Exception:  # pragma: no cover - metrics exporter optional
    class _StubGauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

    Gauge = _StubGauge  # type: ignore

    def start_metrics_server(*args, **kwargs):  # type: ignore
        return None

    synergy_weight_update_alerts_total = _StubGauge()
    synergy_weight_update_failures_total = _StubGauge()
from alert_dispatcher import dispatch_alert

# Prometheus gauges tracking trainer state
synergy_trainer_iterations = Gauge(
    "synergy_trainer_iterations",
    "Total number of synergy training cycles executed",
)
synergy_trainer_last_id = Gauge(
    "synergy_trainer_last_id",
    "ID of the last processed synergy history entry",
)
synergy_trainer_failures_total = Gauge(
    "synergy_trainer_failures_total",
    "Total number of failed synergy training attempts",
)

ALERT_THRESHOLD = int(os.getenv("SYNERGY_WEIGHT_ALERT_THRESHOLD", "5"))

try:
    from . import synergy_weight_cli
except Exception:  # pragma: no cover - optional dependency
    synergy_weight_cli = None  # type: ignore
from . import synergy_history_db as shd


class SynergyWeightCliError(RuntimeError):
    """Raised when ``synergy_weight_cli`` exits with a non-zero status."""

    def __init__(self, code: int) -> None:
        super().__init__(f"synergy_weight_cli exited with code {code}")
        self.code = code


class SynergyAutoTrainer:
    """Periodically train synergy weights from history."""

    def __init__(
        self,
        *,
        history_file: str | Path = "synergy_history.db",
        weights_file: str | Path = "synergy_weights.json",
        interval: float = 600.0,
        progress_file: str | Path = "last_ts.json",
        metrics_port: int | None = None,
    ) -> None:
        self.history_file = Path(history_file)
        self.weights_file = Path(weights_file)
        self.interval = float(interval)
        self.progress_file = Path(progress_file)
        self._last_id = 0
        data_dir = Path(resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")))

        if not self.history_file.exists():
            new_path = data_dir / self.history_file.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.touch()
            logging.getLogger(self.__class__.__name__).warning(
                "history file %s missing - created empty file at %s",
                self.history_file,
                new_path,
            )
            self.history_file = new_path

        weights_path = self.weights_file
        if not weights_path.exists():
            weights_path = data_dir / self.weights_file.name
        lock = FileLock(str(weights_path) + ".lock")
        with lock:
            if not weights_path.exists():
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                weights_path.touch()
                logging.getLogger(self.__class__.__name__).warning(
                    "weights file %s missing - created empty file at %s",
                    self.weights_file,
                    weights_path,
                )
            self.weights_file = weights_path

        if self.progress_file.exists():
            try:
                data = json.loads(self.progress_file.read_text())
                self._last_id = int(data.get("last_id", 0))
            except Exception:
                self._last_id = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_port = metrics_port
        self._metrics_started = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._task: asyncio.Task | None = None

    # --------------------------------------------------------------
    def _load_history(self) -> list[tuple[int, dict[str, float]]]:
        if not self.history_file.exists():
            return []
        try:
            with shd.connect_locked(self.history_file) as conn:
                return shd.fetch_after(conn, self._last_id)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception(
                "failed to load history from %s: %s", self.history_file, exc
            )
            raise

    # --------------------------------------------------------------
    def _train_once(self) -> None:
        hist = self._load_history()
        self.logger.info("processing %d history entries", len(hist))
        if not hist:
            return
        # update metrics for each successful cycle
        try:
            synergy_trainer_iterations.inc()
        except Exception:  # pragma: no cover - metrics may be unavailable
            self.logger.exception(
                "failed to increment synergy_trainer_iterations gauge"
            )
        lock = FileLock(str(self.weights_file) + ".lock")
        success = False
        try:
            with lock:
                synergy_weight_cli.train_from_history(
                    [h[1] for h in hist], self.weights_file
                )
            success = True
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.warning("synergy_weight_cli failed: %s", exc)
            try:
                synergy_trainer_failures_total.inc()
            except Exception:  # pragma: no cover - metrics may be unavailable
                self.logger.exception(
                    "failed to increment synergy_trainer_failures_total gauge"
                )
            try:
                synergy_weight_update_failures_total.inc()
            except Exception:  # pragma: no cover - metrics may be unavailable
                self.logger.exception(
                    "failed to increment synergy_weight_update_failures_total gauge"
                )
            try:
                dispatch_alert(
                    "synergy_weight_update_failure",
                    2,
                    "Weight update failed",
                    {"path": str(self.weights_file)},
                )
                synergy_weight_update_alerts_total.inc()
            except Exception:
                self.logger.exception("failed to dispatch weight alert")
        finally:
            self.logger.info(
                "weight update %s", "succeeded" if success else "failed"
            )
            if success:
                self._last_id = hist[-1][0]
                try:
                    synergy_trainer_last_id.set(float(self._last_id))
                except Exception:  # pragma: no cover - metrics may be unavailable
                    self.logger.exception(
                        "failed to set synergy_trainer_last_id gauge"
                    )
                try:
                    self.progress_file.parent.mkdir(parents=True, exist_ok=True)
                    self.progress_file.write_text(
                        json.dumps({"last_id": self._last_id})
                    )
                    self.logger.info(
                        "progress saved (last_id=%d)", self._last_id
                    )
                except Exception as exc:
                    self.logger.exception(
                        "failed to update progress: %s", exc
                    )

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._train_once()
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.exception("training cycle failed: %s", exc)
            if self._stop.wait(self.interval):
                break

    # --------------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        self.logger.info(
            "starting SynergyAutoTrainer with history=%s weights=%s progress=%s interval=%.1fs metrics_port=%s",
            self.history_file,
            self.weights_file,
            self.progress_file,
            self.interval,
            self.metrics_port,
        )
        if self.metrics_port is not None and not self._metrics_started:
            try:
                self.logger.info(
                    "starting metrics server on port %d", self.metrics_port
                )
                start_metrics_server(int(self.metrics_port))
                self.logger.info(
                    "metrics server running on port %d", self.metrics_port
                )
                self._metrics_started = True
            except Exception:
                self.logger.exception("failed to start metrics server")
        try:
            synergy_trainer_iterations.set(0.0)
            synergy_trainer_last_id.set(float(self._last_id))
        except Exception:
            pass

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self._loop()
                except Exception as exc:  # pragma: no cover - runtime issues
                    self.logger.exception("trainer loop crashed: %s", exc)
                if self._stop.wait(self.interval):
                    break

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self.logger.info("trainer thread started")

    def stop(self) -> None:
        self.logger.info("stopping SynergyAutoTrainer")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
            self.logger.info("trainer thread stopped")

    def restart(self) -> None:
        """Restart the trainer."""
        self.logger.info("restarting SynergyAutoTrainer")
        self.stop()
        self._stop.clear()
        self.start()

    # --------------------------------------------------------------
    async def _async_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._train_once)
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.exception("training cycle failed: %s", exc)
            if await asyncio.to_thread(self._stop.wait, self.interval):
                break

    def start_async(self) -> None:
        if self._task:
            return
        if self.metrics_port is not None and not self._metrics_started:
            try:
                self.logger.info(
                    "starting metrics server on port %d", self.metrics_port
                )
                start_metrics_server(int(self.metrics_port))
                self.logger.info(
                    "metrics server running on port %d", self.metrics_port
                )
                self._metrics_started = True
            except Exception:
                self.logger.exception("failed to start metrics server")
        try:
            synergy_trainer_iterations.set(0.0)
            synergy_trainer_last_id.set(float(self._last_id))
        except Exception:
            pass
        self.logger.info(
            "starting async SynergyAutoTrainer with history=%s weights=%s progress=%s interval=%.1fs metrics_port=%s",
            self.history_file,
            self.weights_file,
            self.progress_file,
            self.interval,
            self.metrics_port,
        )
        self._task = asyncio.create_task(self._async_loop())
        self.logger.info("async trainer task started")

    async def stop_async(self) -> None:
        self.logger.info("stopping SynergyAutoTrainer (async)")
        self._stop.set()
        if self._task:
            await self._task
            self._task = None
            self.logger.info("async trainer task stopped")

    async def restart_async(self) -> None:
        """Restart the trainer within an asyncio loop."""
        self.logger.info("restarting SynergyAutoTrainer (async)")
        await self.stop_async()
        self._stop.clear()
        self.start_async()

# --------------------------------------------------------------
def cli(argv: list[str] | None = None) -> int:
    """Run the auto trainer as a standalone process."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-file",
        default="synergy_history.db",
        help="SQLite history database",
    )
    parser.add_argument(
        "--weights-file",
        default="synergy_weights.json",
        help="Synergy weights JSON file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=600.0,
        help="training interval in seconds",
    )
    parser.add_argument(
        "--progress-file",
        default="last_ts.json",
        help="file tracking the last processed history id",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose Prometheus gauges",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="perform a single training cycle and exit",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="run trainer within asyncio event loop",
    )
    args = parser.parse_args(argv)

    if args.metrics_port is None:
        env_port = os.getenv("SYNERGY_METRICS_PORT") or os.getenv("METRICS_PORT")
        if env_port:
            try:
                args.metrics_port = int(env_port)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Invalid SYNERGY_METRICS_PORT value: %s",
                    env_port,
                )

    trainer = SynergyAutoTrainer(
        history_file=args.history_file,
        weights_file=args.weights_file,
        interval=args.interval,
        progress_file=args.progress_file,
        metrics_port=args.metrics_port,
    )

    if args.run_once:
        if args.metrics_port is not None:
            try:
                logging.getLogger(__name__).info(
                    "starting metrics server on port %d", args.metrics_port
                )
                start_metrics_server(int(args.metrics_port))
                logging.getLogger(__name__).info(
                    "metrics server running on port %d", args.metrics_port
                )
            except Exception as exc:
                logging.getLogger(__name__).error(
                    "failed to start metrics server: %s",
                    exc,
                )
        try:
            synergy_trainer_iterations.set(0.0)
            synergy_trainer_last_id.set(float(trainer._last_id))
        except Exception:
            pass
        rc = 0
        try:
            trainer._train_once()
        except SynergyWeightCliError as exc:
            rc = exc.code or 1
        except Exception:
            rc = 1
        finally:
            if rc:
                try:
                    synergy_trainer_failures_total.inc()
                except Exception:
                    pass
        return rc

    rc = 0
    if args.async_mode:
        async def runner() -> int:
            trainer.start_async()
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                await trainer.stop_async()
            return 0

        rc = asyncio.run(runner())
    else:
        trainer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            trainer.stop()
        rc = 0

    if rc:
        try:
            synergy_trainer_failures_total.inc()
        except Exception:
            pass
    return rc


def main(argv: list[str] | None = None) -> None:
    import sys

    sys.exit(cli(argv))


__all__ = [
    "SynergyAutoTrainer",
    "cli",
    "main",
    "synergy_trainer_iterations",
    "synergy_trainer_last_id",
    "synergy_trainer_failures_total",
    "SynergyWeightCliError",
]
