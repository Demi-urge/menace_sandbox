from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class MessageMetrics:
    length: int
    punct: int
    upper_ratio: float
    time_gap: float


@dataclass
class RunningStats:
    count: int = 0
    length: float = 0.0
    punct: float = 0.0
    upper_ratio: float = 0.0
    time_gap: float = 0.0


class AnomalyDetector:
    """Detect short and long term behavioural anomalies."""

    def __init__(self, short_window: int = 5) -> None:
        self.short_window = short_window
        self._recent: Dict[str, Deque[MessageMetrics]] = {}
        self._baseline: Dict[str, RunningStats] = {}
        self._last_time: Dict[str, float] = {}

    def _extract(self, text: str, timestamp: float, prev_time: float) -> MessageMetrics:
        length = len(text)
        punct = text.count("!") + text.count("?")
        letters = sum(1 for c in text if c.isalpha())
        upper = sum(1 for c in text if c.isupper())
        upper_ratio = upper / max(1, letters)
        time_gap = timestamp - prev_time if prev_time else 0.0
        return MessageMetrics(length, punct, upper_ratio, time_gap)

    def _average(self, records: Deque[MessageMetrics]) -> RunningStats:
        stats = RunningStats()
        if not records:
            return stats
        stats.count = len(records)
        stats.length = sum(m.length for m in records) / stats.count
        stats.punct = sum(m.punct for m in records) / stats.count
        stats.upper_ratio = sum(m.upper_ratio for m in records) / stats.count
        stats.time_gap = sum(m.time_gap for m in records) / stats.count
        return stats

    def _diff(self, m: MessageMetrics, stats: RunningStats) -> float:
        if stats.count == 0:
            return 0.0
        d_len = abs(m.length - stats.length) / (stats.length + 1)
        d_punct = abs(m.punct - stats.punct) / (stats.punct + 1)
        d_upper = abs(m.upper_ratio - stats.upper_ratio) / (stats.upper_ratio + 0.1)
        d_time = abs(m.time_gap - stats.time_gap) / (stats.time_gap + 1)
        return d_len + d_punct + d_upper + d_time

    def detect(self, user_id: str, text: str, *, timestamp: Optional[float] = None) -> float:
        ts = time.time() if timestamp is None else timestamp
        prev = self._last_time.get(user_id, ts)
        metrics = self._extract(text, ts, prev)

        history = self._recent.setdefault(user_id, deque(maxlen=self.short_window))
        short_avg = self._average(history)
        short_score = self._diff(metrics, short_avg)
        history.append(metrics)

        base = self._baseline.setdefault(user_id, RunningStats())
        long_score = self._diff(metrics, base)

        n = base.count
        base.count += 1
        base.length = (base.length * n + metrics.length) / base.count
        base.punct = (base.punct * n + metrics.punct) / base.count
        base.upper_ratio = (base.upper_ratio * n + metrics.upper_ratio) / base.count
        base.time_gap = (base.time_gap * n + metrics.time_gap) / base.count
        self._baseline[user_id] = base
        self._last_time[user_id] = ts

        return (short_score + long_score) / 2
