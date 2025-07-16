"""Simplified scheduler utilities for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Iterable
import logging

logger = logging.getLogger(__name__)


def compute_moving_average(values: List[float], window: int = 3) -> float:
    """Return a simple moving average for ``values``.

    The helper now supports an arbitrary window size and gracefully
    handles an empty input list.  The behaviour for the simple cases
    remains identical to the previous implementation so that existing
    tests continue to pass.
    """

    if not values:
        return 0.0
    window = min(len(values), window)
    return float(sum(values[-window:]) / window)


def is_balolos_clip(clip: dict) -> bool:
    tags = clip.get("tags", [])
    if "balolos" in tags:
        return True
    return clip.get("category") == "balolos"


class Scheduler:
    def __init__(self, clips_file: Path, topics_file: Path, accounts_file: Path, history_file: Path) -> None:
        self.clips_file = Path(clips_file)
        self.topics_file = Path(topics_file)
        self.accounts_file = Path(accounts_file)
        self.history_file = Path(history_file)
        self.accounts = json.loads(self.accounts_file.read_text()).get(
            "accounts", []
        )
        self.clips: Dict[str, Dict[str, Any]] = {}
        self.topics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _load_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return default

    def load(self) -> None:
        """Load clips and topics from disk."""
        self.clips = self._load_json(self.clips_file, {})
        self.topics = self._load_json(self.topics_file, {})

    def _clip_score(self, clip: Dict[str, Any]) -> float:
        """Return priority score for a clip."""

        stats = clip.get("stats", [])
        values = [s.get("views", 0.0) for s in stats if isinstance(s, dict)]
        score = compute_moving_average(values)

        # favour recent clips if a timestamp is present
        created = clip.get("created")
        if created and isinstance(created, str):
            try:
                from datetime import datetime

                age_hours = (
                    datetime.utcnow() - datetime.fromisoformat(created)
                ).total_seconds() / 3600.0
                score *= max(0.1, 24.0 / (age_hours + 1.0))
            except Exception:  # pragma: no cover - invalid timestamp
                logger.error("invalid timestamp '%s'", created)

        if is_balolos_clip(clip):
            score = 0.0
        return float(score)

    def compute_schedule(self, topics: Iterable[str] | None = None) -> List[Dict[str, Any]]:
        """Select the best clip for each account."""
        if not self.clips:
            self.load()
        topics = set(topics or [])
        schedule: List[Dict[str, Any]] = []
        for acc in self.accounts:
            acc_topics = set(acc.get("topics", []))
            if topics:
                acc_topics &= topics

            choices = [
                (cid, clip)
                for cid, clip in self.clips.items()
                if clip.get("topic") in acc_topics
            ]

            if not choices:
                continue

            choices.sort(key=lambda c: self._clip_score(c[1]), reverse=True)
            cid, best = choices[0]
            best.setdefault("scheduled", 0)
            best["scheduled"] += 1
            schedule.append({"account": acc.get("id", acc.get("platform")), "clip": cid})
        return schedule

    def run(self, topics: Iterable[str] | None = None) -> List[Dict[str, Any]]:
        """Generate a posting schedule and store in history file."""
        sched = self.compute_schedule(topics)
        with open(self.history_file, "w", encoding="utf-8") as fh:
            json.dump({"schedule": sched}, fh)
        return sched

__all__ = ["compute_moving_average", "is_balolos_clip", "Scheduler"]
