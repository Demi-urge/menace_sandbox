from __future__ import annotations

"""Prometheus exporter for synergy metrics."""

from pathlib import Path
import json
import logging
import threading
from typing import Dict

from .metrics_exporter import Gauge, start_metrics_server


class SynergyExporter:
    """Periodically read ``synergy_history.json`` and expose latest values."""

    def __init__(
        self,
        history_file: str | Path = "synergy_history.json",
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

    # ------------------------------------------------------------------
    def _load_latest(self) -> Dict[str, float]:
        p = self.history_file
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list) and data:
                entry = data[-1]
                if isinstance(entry, dict):
                    return {str(k): float(v) for k, v in entry.items()}
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("failed to read %s: %s", p, exc)
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
            self._stop.wait(self.interval)

    # ------------------------------------------------------------------
    def start(self) -> None:
        start_metrics_server(self.port)
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        self.logger.info(
            "Synergy metrics exporter running on port %d", self.port
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def start_synergy_exporter(
    *,
    history_file: str | Path = "synergy_history.json",
    interval: float = 5.0,
    port: int = 8003,
) -> SynergyExporter:
    """Start a :class:`SynergyExporter` instance."""

    exp = SynergyExporter(history_file, interval=interval, port=port)
    exp.start()
    return exp


__all__ = ["SynergyExporter", "start_synergy_exporter"]
