from __future__ import annotations

"""Prometheus exporter for synergy metrics."""

from pathlib import Path
import json
import sqlite3
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict

from .metrics_exporter import Gauge, start_metrics_server

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
    def _load_latest(self) -> Dict[str, float]:
        p = self.history_file
        if not p.exists():
            return {}
        try:
            with sqlite3.connect(p) as conn:
                row = conn.execute(
                    "SELECT entry FROM synergy_history ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if row:
                data = json.loads(row[0])
                if isinstance(data, dict):
                    return {str(k): float(v) for k, v in data.items()}
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("failed to read %s: %s", p, exc)
            self.failures += 1
            try:
                exporter_failures.set(float(self.failures))
            except Exception:  # pragma: no cover - metrics library issues
                pass
        return {}

    def _update_loop(self) -> None:
        while not self._stop.is_set():
            vals = self._load_latest()
            for name, value in vals.items():
                g = self._gauges.get(name)
                if g is None:
                    g = Gauge(name, f"Latest value for {name}")
                    self._gauges[name] = g
                try:
                    g.set(float(value))
                except Exception:  # pragma: no cover - metrics library issues
                    self.logger.exception("failed to update gauge %s", name)
                    self.failures += 1
                    try:
                        exporter_failures.set(float(self.failures))
                    except Exception:  # pragma: no cover - metrics library issues
                        pass
            if vals:
                self.last_update = time.time()
            if self.start_time is not None:
                try:
                    exporter_uptime.set(time.time() - self.start_time)
                except Exception:  # pragma: no cover - metrics library issues
                    pass
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
                    {"status": "ok", "updated": exporter.last_update}
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
        start_metrics_server(self.port)
        self.start_time = time.time()
        exporter_uptime.set(0.0)
        exporter_failures.set(float(self.failures))
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        self._start_health_server()
        self.logger.info("Synergy metrics exporter running on port %d", self.port)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.start_time is not None:
            try:
                exporter_uptime.set(time.time() - self.start_time)
            except Exception:  # pragma: no cover - metrics library issues
                pass


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
]
