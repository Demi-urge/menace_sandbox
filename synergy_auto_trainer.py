import json
import logging
import os
import sqlite3
import tempfile
import threading
from pathlib import Path

from . import synergy_weight_cli
from . import synergy_history_db as shd


class SynergyAutoTrainer:
    """Periodically train synergy weights from history."""

    def __init__(
        self,
        *,
        history_file: str | Path = "synergy_history.db",
        weights_file: str | Path = "synergy_weights.json",
        interval: float = 600.0,
    ) -> None:
        self.history_file = Path(history_file)
        self.weights_file = Path(weights_file)
        self.interval = float(interval)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --------------------------------------------------------------
    def _load_history(self) -> list[dict[str, float]]:
        if not self.history_file.exists():
            return []
        try:
            with sqlite3.connect(self.history_file) as conn:
                return shd.fetch_all(conn)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("failed to load history: %s", exc)
            return []

    # --------------------------------------------------------------
    def _train_once(self) -> None:
        hist = self._load_history()
        if not hist:
            return
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        try:
            json.dump(hist, tmp)
            tmp.close()
            try:
                synergy_weight_cli.cli([
                    "--path",
                    str(self.weights_file),
                    "train",
                    tmp.name,
                ])
            except SystemExit:
                pass
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.exception("training failed: %s", exc)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._train_once()
            self._stop.wait(self.interval)

    # --------------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


__all__ = ["SynergyAutoTrainer"]
