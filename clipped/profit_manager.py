"""Profit management utilities used during tests.

This version keeps the basic behaviour relied on by the tests while
extending the logic so that it more closely resembles a real profit
manager.  Existing tests continue to exercise the original behaviour
such as replacing unprofitable clips and limiting accounts per
platform, but additional scoring and topic handling logic has been
implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from .scoutboard import Scoutboard
from .topic_prediction import TopicPredictionEngine


class ProfitManager:
    def __init__(self, clips_file: Path, accounts_file: Path, topics_file: Path, chamber_file: Path) -> None:
        self.clips_file = Path(clips_file)
        self.accounts_file = Path(accounts_file)
        self.topics_file = Path(topics_file)
        self.chamber_file = Path(chamber_file)
        self.scoutboard = Scoutboard(self.clips_file.with_name("scoutboard.json"))

    # ------------------------------------------------------------------
    def _load_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return default

    def _save_json(self, path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    # ------------------------------------------------------------------
    def run(self) -> List[str]:
        clips: Dict[str, Dict[str, Any]] = self._load_json(self.clips_file, {})
        accounts: Dict[str, Any] = self._load_json(
            self.accounts_file, {"accounts": []}
        )
        topics: Dict[str, Any] = self._load_json(self.topics_file, {})
        chamber: Dict[str, Any] = self._load_json(self.chamber_file, {})

        deleted: List[str] = []
        profit_scores = [
            c.get("profit", 0.0) / max(c.get("file_size_mb", 1.0), 1e-6)
            for c in clips.values()
        ]
        avg_score = sum(profit_scores) / len(profit_scores) if profit_scores else 0.0
        threshold = avg_score * 0.5

        next_id = max([int(k) for k in clips.keys()] or [0]) + 1
        for cid, clip in list(clips.items()):
            if clip.get("topic") == "Balolos":
                continue
            score = clip.get("profit", 0.0) / max(clip.get("file_size_mb", 1.0), 1e-6)
            if clip.get("profit", 0.0) <= 0.0 or score < threshold:
                deleted.append(cid)
                del clips[cid]
                clips[str(next_id)] = {
                    "file_size_mb": chamber.get("file_size_mb", 0.0),
                    "profit": 0.0,
                    "topic": clip.get("topic"),
                    "confidence": clip.get("confidence", 0.0),
                    "created": clip.get("created"),
                }
                next_id += 1

        counts: Dict[str, int] = {}
        for acc in accounts.get("accounts", []):
            if acc.get("status") == "removed":
                continue
            plat = acc.get("platform")
            counts[plat] = counts.get(plat, 0) + 1
            if counts[plat] > 200:
                acc["status"] = "removed"

        engine = TopicPredictionEngine()
        suggestions = sorted(
            engine.suggest_topics(list(topics.keys())),
            key=lambda s: s.get("trend_velocity", 0.0),
            reverse=True,
        )
        for s in suggestions:
            name = s.get("name")
            if not name:
                continue
            topics.setdefault(name, {})
            for old in topics:
                if old != name:
                    topics[old].setdefault("hibernated", True)
            for acc in accounts.get("accounts", []):
                acc.setdefault("topics", [])
                if name not in acc["topics"]:
                    acc["topics"].append(name)

        self.scoutboard.update_board(clips)

        self._save_json(self.clips_file, clips)
        self._save_json(self.accounts_file, accounts)
        self._save_json(self.topics_file, topics)
        return deleted

__all__ = ["ProfitManager"]
