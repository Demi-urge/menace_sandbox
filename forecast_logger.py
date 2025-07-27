from __future__ import annotations

"""Append-only logger recording ROI forecasts and threshold details."""

import json
import threading
import time
from pathlib import Path
from typing import Mapping, Any


class ForecastLogger:
    """Write forecast information to a JSON lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, data: Mapping[str, Any]) -> None:
        record = {"timestamp": int(time.time())}
        record.update({k: v for k, v in data.items()})
        line = json.dumps(record)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


__all__ = ["ForecastLogger"]
