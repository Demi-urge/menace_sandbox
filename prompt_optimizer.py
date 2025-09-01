"""Prompt formatting optimisation utilities.

This module analyses prompt experiment logs to discover which formatting
styles yield the best outcomes. Logs are expected to be line delimited JSON
containing at least ``module``, ``action``, ``prompt``, ``success`` and
``roi`` fields. Additional optional fields such as ``coverage`` or
``runtime_improvement`` can be used to weight ROI calculations.

The optimiser groups prompts by structural features – tone, header set and
example placement – and computes success rates as well as weighted ROI
improvements. These aggregated statistics are persisted so that subsequent
runs can build upon previous observations.
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
    header_set: Tuple[str, ...]
    example_placement: str
    has_code: bool
    has_bullets: bool
    success: int = 0
    total: int = 0
    roi_sum: float = 0.0
    weighted_roi_sum: float = 0.0
    weight_sum: float = 0.0

    def update(self, success: bool, roi: float, weight: float) -> None:
        """Update counters with a single observation."""

        self.total += 1
        self.roi_sum += roi
        self.weighted_roi_sum += roi * weight
        self.weight_sum += weight
        if success:
            self.success += 1

    # ------------------------------------------------------------------
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0

    def weighted_roi(self) -> float:
        if self.weight_sum:
            return self.weighted_roi_sum / self.weight_sum
        return self.roi_sum / self.total if self.total else 0.0

    def score(self) -> float:
        """Combined score used to rank configurations."""

        return self.success_rate() * self.weighted_roi()


class PromptOptimizer:
    """Analyse logs and recommend high-performing prompt formats.

    Parameters
    ----------
    log_a, log_b:
        Paths to prompt experiment log files.
    stats_path:
        File used to persist aggregated statistics (JSON format).
    weight_by:
        Optional weighting mode. ``"coverage"`` uses the ``coverage`` field
        of each log entry as the weight, ``"runtime"`` uses the
        ``runtime_improvement`` field and ``None`` applies no additional
        weighting.
    """

    def __init__(
        self,
        log_a: str | Path,
        log_b: str | Path,
        *,
        stats_path: str | Path = "prompt_optimizer_stats.json",
        weight_by: str | None = None,
    ) -> None:
        self.log_paths = [Path(log_a), Path(log_b)]
        self.stats_path = Path(stats_path)
        self.weight_by = weight_by
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
                    self.stats[key] = _Stat(
                        module=item["module"],
                        action=item["action"],
                        tone=item["tone"],
                        header_set=tuple(item.get("header_set", [])),
                        example_placement=item.get("example_placement", "none"),
                        has_code=bool(item.get("has_code")),
                        has_bullets=bool(item.get("has_bullets")),
                        success=int(item.get("success", 0)),
                        total=int(item.get("total", 0)),
                        roi_sum=float(item.get("roi_sum", 0.0)),
                        weighted_roi_sum=float(item.get("weighted_roi_sum", 0.0)),
                        weight_sum=float(item.get("weight_sum", 0.0)),
                    )
                except Exception:
                    continue

    def _key_from_item(self, item: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            item["module"],
            item["action"],
            item["tone"],
            tuple(item.get("header_set", [])),
            item.get("example_placement", "none"),
            bool(item.get("has_code")),
            bool(item.get("has_bullets")),
        )

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _extract_features(self, prompt: str) -> Dict[str, Any]:
        """Derive structural features from ``prompt``."""

        header_matches = list(
            re.finditer(r"^#+\s*(.+)$", prompt, flags=re.MULTILINE)
        )
        headers = [m.group(1).strip() for m in header_matches]
        header_set = tuple(sorted(headers))

        example_positions = [
            m.start() for m in re.finditer(r"Example", prompt, flags=re.IGNORECASE)
        ]
        example_placement: str
        if not example_positions:
            example_placement = "none"
        else:
            half = len(prompt) / 2
            before = [p for p in example_positions if p <= half]
            after = [p for p in example_positions if p > half]
            if before and after:
                example_placement = "mixed"
            elif before:
                example_placement = "start"
            else:
                example_placement = "end"

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
            "header_set": header_set,
            "example_placement": example_placement,
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
            if self.weight_by == "coverage":
                weight = float(entry.get("coverage", 1.0))
            elif self.weight_by == "runtime":
                weight = float(entry.get("runtime_improvement", 1.0))
            else:
                weight = 1.0
            module = entry.get("module", "unknown")
            action = entry.get("action", "unknown")
            feats = self._extract_features(prompt)
            key = (
                module,
                action,
                feats["tone"],
                feats["header_set"],
                feats["example_placement"],
                feats["has_code"],
                feats["has_bullets"],
            )
            stat = self.stats.get(key)
            if not stat:
                stat = _Stat(
                    module=module,
                    action=action,
                    tone=feats["tone"],
                    header_set=feats["header_set"],
                    example_placement=feats["example_placement"],
                    has_code=feats["has_code"],
                    has_bullets=feats["has_bullets"],
                )
                self.stats[key] = stat
            stat.update(success, roi, weight)
        self.persist_statistics()
        return self.stats

    # ------------------------------------------------------------------
    def persist_statistics(self) -> None:
        """Persist aggregated statistics to ``stats_path``."""

        data = []
        for stat in self.stats.values():
            item = asdict(stat)
            # convert tuple to list for JSON serialisation
            item["header_set"] = list(stat.header_set)
            data.append(item)
        self.stats_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def select_format(self, module: str, action: str) -> Dict[str, Any]:
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
            "structured_sections": list(best.header_set),
            "example_placement": best.example_placement,
            "include_code": best.has_code,
            "use_bullets": best.has_bullets,
        }

    # Backwards compatibility ------------------------------------------------
    def suggest_format(self, module: str, action: str) -> Dict[str, Any]:
        """Alias for :meth:`select_format`."""

        return self.select_format(module, action)
