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
    has_system: bool
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

    def score(self, roi_weight: float = 1.0) -> float:
        """Combined score used to rank configurations.

        Parameters
        ----------
        roi_weight:
            Exponent applied to the ROI component when calculating the score.
            Values greater than ``1`` emphasise ROI while values below ``1``
            make success rate more dominant.
        """

        roi = max(self.weighted_roi(), 0.0)
        return self.success_rate() * (roi ** roi_weight)


class PromptOptimizer:
    """Analyse logs and recommend high-performing prompt formats.

    Parameters
    ----------
    success_log, failure_log:
        Paths to success and failure log files respectively. Each file should
        contain one JSON object per line describing an experiment entry. If the
        ``success`` flag is missing it will be inferred from the log type.
    stats_path:
        File used to persist aggregated statistics (JSON format).
    weight_by:
        Optional weighting mode. ``"coverage"`` uses the ``coverage`` field
        of each log entry as the weight, ``"runtime"`` uses the
        ``runtime_improvement`` field and ``None`` applies no additional
        weighting.
    roi_weight:
        Exponent applied to the ROI component when ranking configurations.
    """

    def __init__(
        self,
        success_log: str | Path,
        failure_log: str | Path,
        *,
        stats_path: str | Path = "prompt_optimizer_stats.json",
        weight_by: str | None = None,
        roi_weight: float = 1.0,
    ) -> None:
        self.log_paths = [Path(success_log), Path(failure_log)]
        self.stats_path = Path(stats_path)
        self.weight_by = weight_by
        self.roi_weight = roi_weight
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
                        has_system=bool(item.get("has_system")),
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
            bool(item.get("has_system")),
        )

    # ------------------------------------------------------------------
    def _load_logs(self) -> List[Dict[str, Any]]:
        """Read and parse all configured log files."""

        entries: List[Dict[str, Any]] = []
        for idx, path in enumerate(self.log_paths):
            if not path or not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    # If success flag is missing, infer from log type
                    entry.setdefault("success", idx == 0)
                    entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    def _extract_features(self, prompt: str) -> Dict[str, Any]:
        """Derive structural features from a prompt string."""

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
            has_system = bool(
                entry.get("system")
                or entry.get("system_prompt")
                or entry.get("system_message")
                or (
                    isinstance(entry.get("messages"), list)
                    and any(m.get("role") == "system" for m in entry["messages"])
                )
                or prompt.lstrip().lower().startswith("system:")
            )
            key = (
                module,
                action,
                feats["tone"],
                feats["header_set"],
                feats["example_placement"],
                feats["has_code"],
                feats["has_bullets"],
                has_system,
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
                    has_system=has_system,
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

        suggestions = self.suggest_format(module, action, limit=1)
        return suggestions[0] if suggestions else {}

    def suggest_format(
        self, module: str, action: str, *, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Return top ``limit`` ranked configurations.

        Results are ordered by the combined score of success rate and ROI.
        Each entry includes additional metadata such as success rate and the
        weighted ROI to allow callers to make informed choices.
        """

        ranked = sorted(
            (s for s in self.stats.values() if s.module == module and s.action == action),
            key=lambda s: s.score(self.roi_weight),
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for stat in ranked[:limit]:
            results.append(
                {
                    "tone": stat.tone,
                    "structured_sections": list(stat.header_set),
                    "example_placement": stat.example_placement,
                    "include_code": stat.has_code,
                    "use_bullets": stat.has_bullets,
                    "system_message": stat.has_system,
                    "success_rate": stat.success_rate(),
                    "weighted_roi": stat.weighted_roi(),
                }
            )
        return results

    # ------------------------------------------------------------------
    def refresh(self) -> Dict[Tuple[Any, ...], _Stat]:
        """Rebuild statistics by re-reading the log files."""

        self.stats.clear()
        if self.stats_path.exists():
            self._load_stats()
        return self.aggregate()

DEFAULT_WEIGHTS_PATH = Path("prompt_format_weights.json")


def load_logs(success_path: str | Path, failure_path: str | Path) -> List[Dict[str, Any]]:
    """Aggregate statistics from ``success_path`` and ``failure_path``.

    Parameters
    ----------
    success_path, failure_path:
        Paths to log files containing successful and failed prompt entries.

    Returns
    -------
    list of dict
        Serialised statistics for each unique prompt configuration.
    """

    optimizer = PromptOptimizer(success_path, failure_path, stats_path=DEFAULT_WEIGHTS_PATH)
    return [asdict(stat) for stat in optimizer.stats.values()]


def rank_formats() -> List[Dict[str, Any]]:
    """Return formatting strategies ordered by performance.

    The persisted statistics are loaded from ``prompt_format_weights.json`` and
    ranked using the combined score of success rate and weighted ROI.
    """

    path = DEFAULT_WEIGHTS_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    ranked: List[Dict[str, Any]] = []
    for item in data:
        total = max(int(item.get("total", 0)), 1)
        success = int(item.get("success", 0))
        roi_sum = float(item.get("roi_sum", 0.0))
        weighted_roi_sum = float(item.get("weighted_roi_sum", 0.0))
        weight_sum = float(item.get("weight_sum", 0.0))
        success_rate = success / total
        if weight_sum:
            weighted_roi = weighted_roi_sum / weight_sum
        else:
            weighted_roi = roi_sum / total
        score = success_rate * max(weighted_roi, 0.0)
        ranked.append(
            {
                "headers": item.get("header_set", []),
                "tone": item.get("tone", "neutral"),
                "example_placement": item.get("example_placement", "none"),
                "success_rate": success_rate,
                "weighted_roi": weighted_roi,
                "score": score,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def select_format() -> Dict[str, Any]:
    """Return the single best-performing format configuration."""

    ranked = rank_formats()
    return ranked[0] if ranked else {}

