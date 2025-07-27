from __future__ import annotations

"""Append-only logger recording preset sources and adaptation actions."""

import json
import threading
import time
from pathlib import Path
from typing import Sequence


class PresetLogger:
    """Write preset source information to a JSON lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, run: int, preset_source: str, actions: Sequence[str] | None) -> None:
        record = {
            "timestamp": int(time.time()),
            "run": run,
            "preset_source": preset_source,
            "actions": list(actions) if actions else [],
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


__all__ = ["PresetLogger"]
