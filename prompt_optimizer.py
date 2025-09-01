"""Prompt formatting optimisation utilities.

This module analyses prompt experiment logs to discover which formatting
styles yield the best outcomes. Logs are expected to be line delimited JSON
with at least ``module``, ``action``, ``prompt``, ``success`` and ``roi``
fields. Statistics are aggregated across multiple features such as tone,
header usage and example counts. A ROI weighted success rate ensures that
prompts leading to higher return on investment are preferred.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - allow running without dependency
    SentimentIntensityAnalyzer = None  # type: ignore


@dataclass
class _Stat:
    """Aggregate statistics for a particular prompt configuration."""

    module: str
    action: str
    tone: str
    headers: int
    example_count: int
    has_code: bool
    has_bullets: bool
    success: int = 0
    total: int = 0
    roi_success: float = 0.0
    roi_total: float = 0.0

    def update(self, success: bool, roi: float) -> None:
        """Update counters with a single observation."""

        self.total += 1
        self.roi_total += roi
        if success:
            self.success += 1
            self.roi_success += roi

    def score(self) -> float:
        """Return ROI-weighted success rate for this configuration."""

        if self.roi_total:
            return self.roi_success / self.roi_total
        if self.total:
            return self.success / self.total
        return 0.0


class PromptOptimizer:
    """Analyse logs and suggest high-performing prompt formats."""

    def __init__(
        self,
        log_a: str | Path,
        log_b: str | Path,
        *,
        stats_path: str | Path = "prompt_optimizer_stats.json",
    ) -> None:
        self.log_paths = [Path(log_a), Path(log_b)]
        self.stats_path = Path(stats_path)
        self._sentiment = (
            SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        )
        self.stats: Dict[Tuple[Any, ...], _Stat] = {}
        if self.stats_path.exists():
            self._load_stats()
        # build initial statistics
        self.aggregate()

    # ------------------------------------------------------------------
    def _load_stats(self) -> None:
        """Load persisted statistics if available."""

        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, list):
            for item in data:
                try:
                    key = self._key_from_item(item)
                    self.stats[key] = _Stat(**item)
                except Exception:
                    continue

    def _key_from_item(self, item: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            item["module"],
            item["action"],
            item["tone"],
            int(item["headers"]),
            int(item["example_count"]),
            bool(item.get("has_code")),
            bool(item.get("has_bullets")),
        )

    def _load_logs(self) -> List[Dict[str, Any]]:
        """Read and parse all configured log files."""

        entries: List[Dict[str, Any]] = []
        for path in self.log_paths:
            if not path or not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        continue
        return entries

    def _extract_features(self, prompt: str) -> Dict[str, Any]:
        """Derive structural features from ``prompt``."""

        headers = len(re.findall(r"^#+\s", prompt, flags=re.MULTILINE))
        example_count = len(re.findall(r"Example", prompt, flags=re.IGNORECASE))
        has_code = bool(re.search(r"```", prompt))
        has_bullets = bool(
            re.search(r"^\s*(?:[-*]|\d+\.)\s+", prompt, flags=re.MULTILINE)
        )
        tone = "neutral"
        if self._sentiment:
            try:
                score = self._sentiment.polarity_scores(prompt)["compound"]
                if score > 0.05:
                    tone = "positive"
                elif score < -0.05:
                    tone = "negative"
            except Exception:  # pragma: no cover - best effort
                pass
        return {
            "tone": tone,
            "headers": headers,
            "example_count": example_count,
            "has_code": has_code,
            "has_bullets": has_bullets,
        }

    # ------------------------------------------------------------------
    def aggregate(self) -> Dict[Tuple[Any, ...], _Stat]:
        """Aggregate statistics from all logs and persist them."""

        for entry in self._load_logs():
            prompt = entry.get("prompt", "")
            success = bool(entry.get("success"))
            roi = float(entry.get("roi", 1.0))
            module = entry.get("module", "unknown")
            action = entry.get("action", "unknown")
            feats = self._extract_features(prompt)
            key = (
                module,
                action,
                feats["tone"],
                feats["headers"],
                feats["example_count"],
                feats["has_code"],
                feats["has_bullets"],
            )
            stat = self.stats.get(key)
            if not stat:
                stat = _Stat(
                    module=module,
                    action=action,
                    tone=feats["tone"],
                    headers=feats["headers"],
                    example_count=feats["example_count"],
                    has_code=feats["has_code"],
                    has_bullets=feats["has_bullets"],
                )
                self.stats[key] = stat
            stat.update(success, roi)
        self.persist_statistics()
        return self.stats

    # ------------------------------------------------------------------
    def persist_statistics(self) -> None:
        """Persist aggregated statistics to ``stats_path``."""

        data = [asdict(stat) for stat in self.stats.values()]
        self.stats_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def suggest_format(self, module: str, action: str) -> Dict[str, Any]:
        """Return the best performing configuration for ``module`` and ``action``."""

        best: _Stat | None = None
        best_score = -1.0
        for stat in self.stats.values():
            if stat.module != module or stat.action != action:
                continue
            score = stat.score()
            if score > best_score:
                best_score = score
                best = stat
        if not best:
            return {}
        return {
            "tone": best.tone,
            "headers": best.headers,
            "example_count": best.example_count,
            "has_code": best.has_code,
            "has_bullets": best.has_bullets,
        }
