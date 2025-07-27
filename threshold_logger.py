from __future__ import annotations

"""Append-only logger recording ROI and synergy thresholds."""

import json
import threading
import time
from pathlib import Path
from typing import Optional

class ThresholdLogger:
    """Write threshold values to a JSON lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(
        self,
        run: int,
        roi_threshold: Optional[float],
        synergy_threshold: Optional[float],
        converged: bool,
    ) -> None:
        record = {
            "timestamp": int(time.time()),
            "run": run,
            "roi_threshold": roi_threshold,
            "synergy_threshold": synergy_threshold,
            "converged": bool(converged),
        }
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

__all__ = ["ThresholdLogger"]
