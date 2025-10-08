"""Dashboard and reporting helpers for self‑improvement metrics."""

from __future__ import annotations

import json
import threading
import contextlib
import socket
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - simplified environments
    from menace_sandbox.logging_utils import get_logger
except Exception:  # pragma: no cover - fallback
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

from .data_stores import router
from dynamic_path_router import resolve_path


def load_synergy_history(path: str | Path) -> list[dict[str, float]]:
    """Return synergy history entries from ``path`` SQLite database."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        rows = (
            router.get_connection("synergy_history")
            .execute("SELECT entry FROM synergy_history ORDER BY id")
            .fetchall()
        )
        hist: list[dict[str, float]] = []
        for (text,) in rows:
            data = json.loads(text)
            if isinstance(data, dict):
                hist.append({str(k): float(v) for k, v in data.items()})
        return hist
    except Exception:
        return []


def synergy_stats(history: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return average and variance for each synergy metric."""
    metrics: dict[str, list[float]] = {}
    for entry in history:
        for k, v in entry.items():
            metrics.setdefault(str(k), []).append(float(v))
    stats: dict[str, dict[str, float]] = {}
    for name, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        stats[name] = {
            "average": float(arr.mean()) if arr.size else 0.0,
            "variance": float(arr.var()) if arr.size else 0.0,
        }
    return stats


def synergy_ma(
    history: list[dict[str, float]], window: int = 5
) -> list[dict[str, float]]:
    """Return rolling averages over ``window`` samples for each metric."""
    if window < 1:
        raise ValueError("window must be positive")
    metrics = sorted({k for d in history for k in d})
    ma_history: list[dict[str, float]] = []
    for idx in range(len(history)):
        ma_entry: dict[str, float] = {}
        start = max(0, idx + 1 - window)
        for name in metrics:
            vals = [history[j].get(name, 0.0) for j in range(start, idx + 1)]
            arr = np.array(vals, dtype=float)
            ma_entry[name] = float(arr.mean()) if arr.size else 0.0
        ma_history.append(ma_entry)
    return ma_history


class SynergyDashboard:
    """Expose synergy metrics via a small Flask app."""

    def __init__(
        self,
        history_file: str | Path = "synergy_history.db",
        *,
        ma_window: int = 5,
        exporter_host: str | None = None,
        exporter_port: int = 8003,
        refresh_interval: float = 5.0,
        max_history: int | None = None,
        metrics_timeout: float = 1.0,
    ) -> None:
        from flask import Flask, jsonify  # type: ignore

        self.logger = get_logger(self.__class__.__name__)
        self.history_file = Path(resolve_path(history_file))
        self.ma_window = ma_window
        self.exporter_host = exporter_host
        self.exporter_port = exporter_port
        self.refresh_interval = float(refresh_interval)
        self.max_history = max_history
        self.metrics_timeout = float(metrics_timeout)
        self._history: list[dict[str, float]] = []
        self._last_metrics: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.jsonify = jsonify
        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/stats", "stats", self.stats)
        self.app.add_url_rule("/plot.png", "plot", self.plot)
        self.app.add_url_rule("/history", "history", self.history)

        if self.exporter_host:
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

    def _load(self) -> list[dict[str, float]]:
        if self.exporter_host:
            return list(self._history)
        return load_synergy_history(self.history_file)

    def _parse_metrics(self, text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, value = parts
            if name.startswith("synergy_"):
                try:
                    metrics[name] = float(value)
                except ValueError:
                    continue
        return metrics

    def _fetch_exporter_metrics(self) -> dict[str, float]:
        try:
            import requests  # type: ignore
        except ImportError:  # pragma: no cover - missing dependency
            self.logger.warning("requests package missing; cannot fetch metrics")
            return dict(self._last_metrics) if self._last_metrics else {}

        url = f"http://{self.exporter_host}:{self.exporter_port}/metrics"
        try:
            resp = requests.get(url, timeout=self.metrics_timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"status {resp.status_code}")
            metrics = self._parse_metrics(resp.text)
            if metrics:
                self._last_metrics = metrics
            return metrics
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.warning("failed to fetch metrics from %s: %s", url, exc)
            return dict(self._last_metrics) if self._last_metrics else {}

    def _update_loop(self) -> None:
        while not self._stop.is_set():
            vals = self._fetch_exporter_metrics()
            if vals:
                self._history.append(vals)
                if self.max_history and len(self._history) > self.max_history:
                    self._history = self._history[-self.max_history :]
            self._stop.wait(self.refresh_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def index(self) -> tuple[str, int]:
        return "Synergy dashboard running. Access /stats", 200

    def stats(self) -> tuple[str, int]:
        hist = self._load()
        ma_hist = synergy_ma(hist, self.ma_window)
        data = {
            "stats": synergy_stats(hist),
            "latest": hist[-1] if hist else {},
            "rolling_average": ma_hist[-1] if ma_hist else {},
        }
        return self.jsonify(data), 200

    def history(self) -> tuple[list[dict[str, float]], int]:
        return self.jsonify(self._load()), 200

    def plot(self) -> tuple[bytes, int, dict[str, str]]:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        history = self._load()
        ma_history = synergy_ma(history, self.ma_window)
        metrics = sorted({k for d in history for k in d})
        labels = list(range(len(history)))
        fig, ax = plt.subplots()
        for name in metrics:
            vals = [d.get(name, 0.0) for d in history]
            ax.plot(labels, vals, label=name)
            ma_vals = [d.get(name, 0.0) for d in ma_history]
            ax.plot(labels, ma_vals, label=f"{name}_ma", linestyle="--")
        if metrics:
            ax.legend()
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 5001,
        *,
        wsgi: str = "flask",
    ) -> None:
        """Run the dashboard using the selected WSGI/ASGI server."""
        with contextlib.closing(socket.socket()) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                self.logger.error("port %d in use", port)
                raise

        server = wsgi.lower()
        if server == "flask":
            self.app.run(host=host, port=port)
            return

        if server == "gunicorn":
            try:
                from gunicorn.app.base import BaseApplication  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("gunicorn is required for --wsgi gunicorn") from exc

            class _GunicornApp(BaseApplication):
                def __init__(self, app):
                    self.application = app
                    super().__init__()

                def load_config(self):  # pragma: no cover - runtime setup
                    self.cfg.set("bind", f"{host}:{port}")
                    self.cfg.set("workers", 1)

                def load(self):  # pragma: no cover - runtime setup
                    return self.application

            _GunicornApp(self.app).run()
            return

        if server == "uvicorn":
            try:
                import uvicorn  # type: ignore
                from starlette.middleware.wsgi import WSGIMiddleware
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("uvicorn is required for --wsgi uvicorn") from exc

            uvicorn.run(WSGIMiddleware(self.app), host=host, port=port, workers=1)
            return


__all__ = [
    "SynergyDashboard",
    "load_synergy_history",
    "synergy_stats",
    "synergy_ma",
]

