"""Minimal Scoutboard implementation for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


class Scoutboard:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def update_board(self, clips: Dict[str, Dict[str, Any]]) -> None:
        board = {}
        for clip in clips.values():
            topic = clip.get("topic", "unknown")
            entry = board.setdefault(topic, {"history": []})
            size = max(clip.get("file_size_mb", 1.0), 1e-6)
            profit = clip.get("profit", 0.0)
            entry["history"].append({"profit_per_mb": profit / size})
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(board, fh)

__all__ = ["Scoutboard"]
