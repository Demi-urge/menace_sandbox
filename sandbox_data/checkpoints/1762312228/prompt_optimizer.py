"""Prompt formatting optimisation utilities.

This module analyses prompt experiment logs to discover which formatting
styles yield the best outcomes. Logs are expected to be line delimited JSON
containing at least ``module``, ``action``, ``prompt`` (or ``prompt_text``),
``success`` and ``roi`` fields. The ``prompt`` field may be a flat string or a
structured object with ``system``/``user`` parts. Additional optional fields
such as ``coverage`` or ``runtime_improvement`` can be used to weight ROI
calculations.

The optimiser groups prompts by structural features – tone, header set and
example placement – and computes success rates, weighted ROI improvements and
average runtime deltas. These aggregated statistics are persisted so that
subsequent runs can build upon previous observations.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dynamic_path_router import resolve_path

try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - allow running without dependency
    SentimentIntensityAnalyzer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from failure_fingerprint_store import FailureFingerprintStore
    from failure_fingerprint import FailureFingerprint
except Exception:  # pragma: no cover - allow running without dependency
    FailureFingerprintStore = None  # type: ignore
    FailureFingerprint = None  # type: ignore


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

    runtime_improvement_sum: float = 0.0
    penalty_factor: float = 1.0

    def update(
        self,
        success: bool,
        roi: float,
        weight: float,
        runtime_improvement: float | None = None,
    ) -> None:
        """Update counters with a single observation."""

        self.total += 1
        self.roi_sum += roi
        self.weighted_roi_sum += roi * weight
        self.weight_sum += weight
        if runtime_improvement is not None:
            self.runtime_improvement_sum += runtime_improvement
        if success:
            self.success += 1

    # ------------------------------------------------------------------
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0

    def weighted_roi(self) -> float:
        if self.weight_sum:
            return self.weighted_roi_sum / self.weight_sum
        return self.roi_sum / self.total if self.total else 0.0

    def avg_runtime_improvement(self) -> float:
        return (
            self.runtime_improvement_sum / self.total if self.total else 0.0
        )

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
        base = self.success_rate() * (roi ** roi_weight)
        return base * self.penalty_factor


@dataclass
class _StrategyStat:
    """Aggregate statistics for a particular prompt strategy."""

    strategy: str
    success: int = 0
    total: int = 0
    roi_sum: float = 0.0
    weighted_roi_sum: float = 0.0
    weight_sum: float = 0.0

    def update(self, success: bool, roi: float, weight: float) -> None:
        self.total += 1
        self.roi_sum += roi
        self.weighted_roi_sum += roi * weight
        self.weight_sum += weight
        if success:
            self.success += 1

    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0

    def weighted_roi(self) -> float:
        if self.weight_sum:
            return self.weighted_roi_sum / self.weight_sum
        return self.roi_sum / self.total if self.total else 0.0

    def score(self, roi_weight: float = 1.0) -> float:
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
    failure_store:
        Optional :class:`FailureFingerprintStore` instance providing access to
        previously recorded failure fingerprints. When supplied, prompts that
        belong to high-frequency failure clusters receive a score penalty.
        failure_fingerprints_path:
        Optional path to a ``failure_fingerprints.jsonl`` file. The fingerprints
        contained in this file are treated as additional failure log entries and
        are also used to apply score penalties when ``failure_store`` is
        unavailable.
    fingerprint_threshold:
        Minimum number of fingerprints before a penalty is applied.
    """

    def __init__(
        self,
        success_log: str | Path,
        failure_log: str | Path,
        *,
        stats_path: str | Path = "prompt_optimizer_stats.json",
        strategy_path: str | Path = "_strategy_stats.json",
        weight_by: str | None = None,
        roi_weight: float = 1.0,
        failure_store: FailureFingerprintStore | None = None,
        failure_fingerprints_path: str | Path | None = None,
        fingerprint_threshold: int = 3,
    ) -> None:
        root = resolve_path(".")

        def _resolve(p: str | Path) -> Path:
            try:
                return resolve_path(str(p))
            except FileNotFoundError:
                return root / Path(p)

        self.log_paths = [_resolve(success_log), _resolve(failure_log)]
        if failure_fingerprints_path:
            ff_path = _resolve(failure_fingerprints_path)
            self.log_paths.append(ff_path)
            self.failure_fingerprints_path = ff_path
        else:
            self.failure_fingerprints_path = None
        self.stats_path = _resolve(stats_path)
        self.strategy_path = _resolve(strategy_path)
        self.weight_by = weight_by
        self.roi_weight = roi_weight
        self.failure_store = failure_store
        self.fingerprint_threshold = fingerprint_threshold
        self._sentiment = (
            SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        )
        self.stats: Dict[Tuple[Any, ...], _Stat] = {}
        self.strategy_stats: Dict[str, _StrategyStat] = {}
        if self.stats_path.exists():
            self._load_stats()
        if self.strategy_path.exists():
            self._load_strategy_stats()
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
                        runtime_improvement_sum=float(
                            item.get("runtime_improvement_sum", 0.0)
                        ),
                        penalty_factor=float(item.get("penalty_factor", 1.0)),
                    )
                except Exception:
                    continue

    # ------------------------------------------------------------------
    def _load_strategy_stats(self) -> None:
        """Load persisted per-strategy statistics if available."""

        try:
            data = json.loads(self.strategy_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    self.strategy_stats[str(k)] = _StrategyStat(
                        strategy=str(k),
                        success=int(v.get("success", 0)),
                        total=int(v.get("total", 0)),
                        roi_sum=float(v.get("roi_sum", 0.0)),
                        weighted_roi_sum=float(v.get("weighted_roi_sum", 0.0)),
                        weight_sum=float(v.get("weight_sum", 0.0)),
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
                    if "module" not in entry and "filename" in entry:
                        entry["module"] = entry.get("filename")
                    if "action" not in entry and (
                        "function" in entry or "function_name" in entry
                    ):
                        entry["action"] = entry.get("function") or entry.get(
                            "function_name"
                        )
                    if "prompt" not in entry and "prompt_text" in entry:
                        entry["prompt"] = entry.get("prompt_text")
                    # If success flag is missing, infer from log type
                    entry.setdefault("success", idx == 0)
                    entry.setdefault("failure_reason", None)
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
            prompt_data = entry.get("prompt", "")
            if isinstance(prompt_data, dict):
                parts = [
                    prompt_data.get("system", ""),
                    *prompt_data.get("examples", []),
                    prompt_data.get("user", ""),
                ]
                prompt = "\n".join([p for p in parts if p])
            else:
                prompt = str(prompt_data)
            if not prompt and isinstance(entry.get("prompt_text"), str):
                prompt = entry["prompt_text"]
            success = bool(entry.get("success"))
            roi_field = entry.get("roi", 0.0)
            coverage_val = None
            runtime_val = None
            if isinstance(roi_field, dict):
                roi = float(roi_field.get("roi_delta", 0.0))
                coverage_val = roi_field.get("coverage")
                runtime_val = roi_field.get("runtime_improvement")
            else:
                roi = float(roi_field)
                coverage_val = entry.get("coverage")
                runtime_val = entry.get("runtime_improvement")
            if self.weight_by == "coverage":
                weight = float(coverage_val if coverage_val is not None else 1.0)
            elif self.weight_by == "runtime":
                weight = float(runtime_val if runtime_val is not None else 1.0)
            else:
                weight = 1.0
            module = entry.get("module", "unknown")
            action = entry.get("action", "unknown")
            strategy = entry.get("strategy") or entry.get("prompt_id")
            feats = self._extract_features(prompt)
            has_system = bool(
                entry.get("system")
                or entry.get("system_prompt")
                or entry.get("system_message")
                or (
                    isinstance(entry.get("messages"), list)
                    and any(m.get("role") == "system" for m in entry["messages"])
                )
                or (isinstance(prompt_data, dict) and prompt_data.get("system"))
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
            stat.update(success, roi, weight, runtime_val)
            if strategy:
                sstat = self.strategy_stats.get(str(strategy))
                if not sstat:
                    sstat = _StrategyStat(strategy=str(strategy))
                    self.strategy_stats[str(strategy)] = sstat
                sstat.update(success, roi, weight)
        self._apply_cluster_penalties()
        self.persist_statistics()
        self.persist_strategy_statistics()
        return self.stats

    # ------------------------------------------------------------------
    def _cluster_stats_from_file(self, path: Path) -> Dict[Any, Dict[str, Any]]:
        """Return cluster statistics parsed from a JSONL fingerprint log."""

        clusters: Dict[Any, Dict[str, Any]] = {}
        try:
            fh = path.open("r", encoding="utf-8")
        except Exception:
            return clusters
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                cid = data.get("cluster_id") or data.get("hash") or data.get(
                    "prompt_text"
                )
                info = clusters.setdefault(cid, {"size": 0, "example": None})
                info["size"] += int(data.get("count", 1))
                if info["example"] is None and FailureFingerprint is not None:
                    try:
                        info["example"] = FailureFingerprint(**data)
                    except Exception:
                        info["example"] = data
        return clusters

    def _apply_cluster_penalties(self) -> None:
        """Apply score penalties based on failure fingerprint clusters."""

        clusters: Dict[Any, Dict[str, Any]] = {}
        if self.failure_store is not None:
            try:
                clusters = self.failure_store.cluster_stats()
            except Exception:  # pragma: no cover - best effort
                clusters = {}
        elif self.failure_fingerprints_path and self.failure_fingerprints_path.exists():
            clusters = self._cluster_stats_from_file(self.failure_fingerprints_path)

        for info in clusters.values():
            fp = info.get("example")
            if not fp:
                continue
            prompt = getattr(fp, "prompt_text", "")
            if not prompt:
                continue
            module = getattr(fp, "filename", "unknown")
            action = getattr(fp, "function", getattr(fp, "function_name", "unknown"))
            feats = self._extract_features(prompt)
            has_system = prompt.lstrip().lower().startswith("system:")
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
            size = int(info.get("size", 0))
            stat.total += size
            stat.success = max(0, stat.success - size)
            if size > self.fingerprint_threshold:
                penalty = size - self.fingerprint_threshold
                stat.penalty_factor *= 1 / (1 + penalty)

    # ------------------------------------------------------------------
    def persist_statistics(self) -> None:
        """Persist aggregated statistics to ``stats_path``."""

        data = []
        for stat in self.stats.values():
            item = asdict(stat)
            # convert tuple to list for JSON serialisation
            item["header_set"] = list(stat.header_set)
            item["score"] = stat.score(self.roi_weight)
            data.append(item)
        self.stats_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def persist_strategy_statistics(self) -> None:
        """Persist per-strategy statistics to ``strategy_path``."""

        data: Dict[str, Dict[str, float]] = {}
        for stat in self.strategy_stats.values():
            data[stat.strategy] = {
                "success": stat.success,
                "total": stat.total,
                "roi_sum": stat.roi_sum,
                "weighted_roi_sum": stat.weighted_roi_sum,
                "weight_sum": stat.weight_sum,
            }
        self.strategy_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
                    "avg_runtime_improvement": stat.avg_runtime_improvement(),
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


DEFAULT_WEIGHTS_PATH = resolve_path("prompt_format_weights.json")
DEFAULT_STRATEGY_PATH = resolve_path("_strategy_stats.json")


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
        penalty_factor = float(item.get("penalty_factor", 1.0))
        stored_score = item.get("score")
        if stored_score is not None:
            score = float(stored_score)
        else:
            score = success_rate * max(weighted_roi, 0.0) * penalty_factor
        ranked.append(
            {
                "headers": item.get("header_set", []),
                "tone": item.get("tone", "neutral"),
                "example_placement": item.get("example_placement", "none"),
                "success_rate": success_rate,
                "weighted_roi": weighted_roi,
                "score": score,
                "penalty_factor": penalty_factor,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def load_strategy_stats(path: str | Path | None = None) -> Dict[str, Dict[str, float]]:
    """Return per-strategy statistics including scores."""

    from self_improvement.prompt_strategy_manager import PromptStrategyManager

    p = path if path is not None else DEFAULT_STRATEGY_PATH
    return PromptStrategyManager.load_strategy_stats(p)


def select_strategy(path: str | Path | None = None) -> str | None:
    """Return the strategy with the highest combined score."""

    stats = load_strategy_stats(path)
    best: str | None = None
    best_score = -1.0
    for strat, rec in stats.items():
        score = rec.get("score", 0.0)
        if score > best_score:
            best = strat
            best_score = score
    return best


def select_format() -> Dict[str, Any]:
    """Return the single best-performing format configuration."""

    ranked = rank_formats()
    return ranked[0] if ranked else {}
