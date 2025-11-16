from __future__ import annotations

"""Utility for running SynergyExporter and SynergyAutoTrainer together."""

import argparse
import atexit
import contextlib
import os
import signal
import socket
import sys
import time
from pathlib import Path

if os.getenv("SANDBOX_CENTRAL_LOGGING") is None:
    os.environ["SANDBOX_CENTRAL_LOGGING"] = "1"

from menace.synergy_exporter import SynergyExporter
from menace.synergy_auto_trainer import SynergyAutoTrainer
from menace.audit_trail import AuditTrail
from logging_utils import get_logger, setup_logging
from synergy_monitor import ExporterMonitor
from menace.metrics_exporter import start_metrics_server
from dynamic_path_router import resolve_path


logger = get_logger(__name__)


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


# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run exporter and auto trainer according to environment variables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox-data-dir",
        default=resolve_path("sandbox_data"),
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
    metrics_port = int(os.getenv("SYNERGY_METRICS_PORT", "8003"))
    if os.getenv("EXPORT_SYNERGY_METRICS") == "1":
        port = metrics_port
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
        if exporter is None:
            port = metrics_port
            if not _port_available(port):
                logger.error("synergy metrics port %d in use", port)
                port = _free_port()
                logger.info("using port %d for synergy metrics server", port)
            start_metrics_server(port)
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
        logger.warning("synergy tools interrupted")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
