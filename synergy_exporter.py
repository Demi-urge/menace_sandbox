from __future__ import annotations

"""Prometheus exporter for synergy metrics."""

from pathlib import Path
import json
import sqlite3
import logging
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict

from .metrics_exporter import Gauge, start_metrics_server, stop_metrics_server
try:  # pragma: no cover - optional dependency
    from . import synergy_auto_trainer  # ensure trainer gauges are registered
except Exception:  # pragma: no cover - optional
    synergy_auto_trainer = None
try:  # pragma: no cover - optional relative/absolute
    from .retry_utils import with_retry
except Exception:  # pragma: no cover - package aliasing
    from retry_utils import with_retry  # type: ignore


# Gauges tracking exporter uptime and failures
exporter_uptime = Gauge(
    "synergy_exporter_uptime_seconds",
    "Uptime of the SynergyExporter in seconds",
)
exporter_failures = Gauge(
    "synergy_exporter_failures_total",
    "Total number of exporter update failures",
)


class SynergyExporter:
    """Periodically read ``synergy_history.db`` and expose latest values."""

    def __init__(
        self,
        history_file: str | Path = "synergy_history.db",
        *,
        interval: float = 5.0,
        port: int = 8003,
    ) -> None:
        self.history_file = Path(history_file)
        self.interval = float(interval)
        self.port = int(port)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gauges: Dict[str, Gauge] = {}
        self._health_server: HTTPServer | None = None
        self._health_thread: threading.Thread | None = None
        self.health_port: int | None = None
        self.last_update: float | None = None
        self.start_time: float | None = None
        self.failures = 0

    # ------------------------------------------------------------------
    def _safe_set_gauge(
        self, gauge: Gauge, value: float, *, count_failure: bool = False
    ) -> None:
        def op() -> None:
            gauge.set(value)

        try:
            with_retry(op, attempts=3, delay=0.1, logger=self.logger)
        except Exception:  # pragma: no cover - metrics library issues
            self.logger.exception("failed to update gauge %s", getattr(gauge, "name", ""))
            if count_failure:
                self.failures += 1
                # best effort to record failure without infinite recursion
                self._safe_set_gauge(exporter_failures, float(self.failures))

    def _load_latest(self) -> Dict[str, float]:
        p = self.history_file
        if not p.exists():
            return {}
        try:
            with sqlite3.connect(self.history_file) as conn:  # noqa: P204
                row = conn.execute(
                    "SELECT entry FROM synergy_history ORDER BY id DESC LIMIT 1"
                ).fetchone()
        except sqlite3.Error as exc:
            self.logger.exception("SQLite error reading %s: %s", p, exc)
            raise
        if not row:
            return {}
        try:
            data = json.loads(row[0])
        except json.JSONDecodeError as exc:
            self.logger.exception("invalid JSON in %s: %s", p, exc)
            raise
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
        return {}

    def _update_loop(self) -> None:
        while not self._stop.is_set():
            try:
                vals = self._load_latest()
            except sqlite3.Error as exc:
                self.logger.error("SQLite error during update: %s", exc)
                self.failures += 1
                self._safe_set_gauge(exporter_failures, float(self.failures))
                self._stop.wait(self.interval)
                continue
            except json.JSONDecodeError as exc:
                self.logger.error("JSON error during update: %s", exc)
                self.failures += 1
                self._safe_set_gauge(exporter_failures, float(self.failures))
                self._stop.wait(self.interval)
                continue
            for name, value in vals.items():
                g = self._gauges.get(name)
                if g is None:
                    g = Gauge(name, f"Latest value for {name}")
                    self._gauges[name] = g
                self._safe_set_gauge(g, float(value), count_failure=True)
            if vals:
                self.last_update = time.time()
            if self.start_time is not None:
                self._safe_set_gauge(exporter_uptime, time.time() - self.start_time)
            self._stop.wait(self.interval)

    def _start_health_server(self) -> None:
        """Launch a minimal HTTP endpoint returning exporter status."""
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # type: ignore[override]
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(
                    {
                        "healthy": exporter.failures == 0,
                        "last_update": exporter.last_update,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:  # pragma: no cover - silence
                return

        server = HTTPServer(("0.0.0.0", 0), Handler)
        self.health_port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._health_server = server
        self._health_thread = thread

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.logger.info(
            "starting SynergyExporter with history=%s port=%d interval=%.1fs",
            self.history_file,
            self.port,
            self.interval,
        )
        self.logger.info("starting metrics server on port %d", self.port)
        start_metrics_server(self.port)
        self.logger.info("metrics server running on port %d", self.port)
        self.start_time = time.time()
        self._safe_set_gauge(exporter_uptime, 0.0)
        self._safe_set_gauge(exporter_failures, float(self.failures))
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        self._start_health_server()
        self.logger.info("Synergy metrics exporter running on port %d", self.port)

    def stop(self) -> None:
        self.logger.info("stopping SynergyExporter")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._health_server:
            try:
                self._health_server.shutdown()
                self._health_server.server_close()
            except Exception:  # pragma: no cover - runtime issues
                self.logger.exception("failed to stop health server")
            if self._health_thread:
                self._health_thread.join(timeout=1.0)
            self._health_server = None
            self._health_thread = None
        if self.start_time is not None:
            self._safe_set_gauge(exporter_uptime, time.time() - self.start_time)
        stop_metrics_server()

    def restart(self) -> None:
        """Restart the exporter."""
        self.logger.info("restarting SynergyExporter")
        self.stop()
        self._stop.clear()
        self.start()


def start_synergy_exporter(
    *,
    history_file: str | Path = "synergy_history.db",
    interval: float = 5.0,
    port: int = 8003,
) -> SynergyExporter:
    """Start a :class:`SynergyExporter` instance."""

    exp = SynergyExporter(history_file, interval=interval, port=port)
    exp.start()
    return exp


__all__ = [
    "SynergyExporter",
    "start_synergy_exporter",
    "exporter_uptime",
    "exporter_failures",
    "cli",
    "main",
]


def cli(argv: list[str] | None = None) -> int:
    """Run the exporter as a standalone process."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-file",
        default="synergy_history.db",
        help="SQLite history database",
    )
    parser.add_argument("--port", type=int, default=8003, help="metrics port")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="update interval in seconds",
    )
    args = parser.parse_args(argv)

    exporter = start_synergy_exporter(
        history_file=args.history_file, port=args.port, interval=args.interval
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        exporter.stop()
    return 0


def main(argv: list[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
