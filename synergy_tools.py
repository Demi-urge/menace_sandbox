from __future__ import annotations

"""Utility for running SynergyExporter and SynergyAutoTrainer together."""

import argparse
import atexit
import contextlib
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

from menace.synergy_exporter import SynergyExporter
from menace.synergy_auto_trainer import SynergyAutoTrainer
from menace.audit_trail import AuditTrail
from logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


EXPORTER_CHECK_INTERVAL = float(os.getenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "10"))


def _port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if the TCP ``port`` is free on ``host``."""
    with contextlib.closing(socket.socket()) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _free_port() -> int:
    """Return an available TCP port."""
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _exporter_health_ok(exp: SynergyExporter, *, max_age: float = 30.0) -> bool:
    """Return ``True`` if the exporter health endpoint responds and is fresh."""
    if exp._thread is None or not exp._thread.is_alive():
        return False
    if exp.health_port is None:
        return False
    try:
        with urllib.request.urlopen(
            f"http://localhost:{exp.health_port}/health", timeout=3
        ) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode())
            updated = data.get("updated")
            if updated is None:
                return False
            if time.time() - float(updated) > max_age:
                return False
    except Exception:
        return False
    return True


class ExporterMonitor:
    """Background monitor to keep the exporter running."""

    def __init__(
        self,
        exporter: SynergyExporter,
        log: AuditTrail,
        *,
        interval: float = EXPORTER_CHECK_INTERVAL,
    ) -> None:
        self.exporter = exporter
        self.log = log
        self.interval = float(interval)
        self.restart_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.exporter.stop()
        except Exception:
            logger.exception("failed to stop synergy exporter")

    # ------------------------------------------------------------------
    def _restart(self) -> None:
        try:
            self.exporter.stop()
        except Exception:
            logger.exception("failed to stop synergy exporter")
        try:
            self.exporter = SynergyExporter(
                history_file=str(self.exporter.history_file),
                interval=self.exporter.interval,
                port=self.exporter.port,
            )
            self.exporter.start()
            self.restart_count += 1
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_restarted",
                    "restart_count": self.restart_count,
                }
            )
        except Exception as exc:
            logger.warning("failed to restart synergy exporter: %s", exc)
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_restart_failed",
                    "error": str(exc),
                }
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not _exporter_health_ok(self.exporter):
                self._restart()
            self._stop.wait(self.interval)


# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run exporter and auto trainer according to environment variables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox-data-dir",
        default="sandbox_data",
        help="directory containing synergy data files",
    )
    args = parser.parse_args(argv)

    setup_logging()

    data_dir = Path(args.sandbox_data_dir)
    cleanup_funcs: list[callable] = []

    def _cleanup() -> None:
        for func in cleanup_funcs:
            try:
                func()
            except Exception:
                logger.exception("cleanup failed")

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: (_cleanup(), sys.exit(0)))

    exporter: SynergyExporter | None = None
    monitor: ExporterMonitor | None = None
    if os.getenv("EXPORT_SYNERGY_METRICS") == "1":
        port = int(os.getenv("SYNERGY_METRICS_PORT", "8003"))
        if not _port_available(port):
            logger.error("synergy exporter port %d in use", port)
            port = _free_port()
            logger.info("using port %d for synergy exporter", port)
        history_file = data_dir / "synergy_history.db"
        exporter = SynergyExporter(history_file=str(history_file), port=port)
        try:
            exporter.start()
            monitor = ExporterMonitor(exporter, AuditTrail(str(data_dir / "synergy_tools.log")))
            monitor.start()
            cleanup_funcs.append(monitor.stop)
        except Exception:
            logger.exception("failed to start synergy exporter")

    trainer: SynergyAutoTrainer | None = None
    if os.getenv("AUTO_TRAIN_SYNERGY") == "1":
        try:
            interval = float(os.getenv("AUTO_TRAIN_INTERVAL", "600"))
        except Exception:
            interval = 600.0
        history_file = data_dir / "synergy_history.db"
        weights_file = data_dir / "synergy_weights.json"
        trainer = SynergyAutoTrainer(
            history_file=str(history_file),
            weights_file=str(weights_file),
            interval=interval,
        )
        try:
            trainer.start()
            cleanup_funcs.append(trainer.stop)
        except Exception:
            logger.exception("failed to start synergy auto trainer")

    if not cleanup_funcs:
        logger.info("nothing to do; enable EXPORT_SYNERGY_METRICS or AUTO_TRAIN_SYNERGY")
        return

    logger.info("synergy tools running - press Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
