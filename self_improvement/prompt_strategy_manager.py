"""Unified prompt strategy rotation and tracking utilities."""

from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import time
import math
from typing import Any, Callable, Dict, Sequence

from filelock import FileLock

from dynamic_path_router import resolve_path


DEFAULT_STRATEGIES: list[str] = [
    "strict_fix",
    "delete_rebuild",
    "comment_refactor",
    "unit_test_rewrite",
]

KEYWORD_MAP: Dict[str, str] = {
    "score_drop": "strict_fix",
    "tests_failed": "unit_test_rewrite",
    "entropy_regression": "comment_refactor",
    "test": "unit_test_rewrite",
    "comment": "comment_refactor",
    "refactor": "strict_fix",
    "delete": "delete_rebuild",
    "rebuild": "delete_rebuild",
}


class PromptStrategyManager:
    """Maintain prompt strategy rotation state and performance metrics.

    The manager consolidates previous rotation helpers by persisting the
    current strategy index, tracking consecutive failures, mapping failure
    reasons to strategies and recording ROI statistics for each strategy.
    """

    def __init__(
        self,
        strategies: Sequence[str] | None = None,
        state_path: Path | str | None = None,
        keyword_map: Dict[str, str] | None = None,
        stats_path: Path | str | None = None,
    ) -> None:
        self.strategies: list[str] = list(strategies or DEFAULT_STRATEGIES)
        if state_path is None:
            try:  # local import to avoid heavy dependency during import
                from .init import _data_dir

                state_path = _data_dir() / "prompt_strategy_state.json"
            except Exception:  # pragma: no cover - fallback when init unavailable
                state_path = resolve_path("prompt_strategy_state.json")
        if not isinstance(state_path, Path):
            state_path = resolve_path(state_path)
        self.state_path = state_path
        if stats_path is None:
            stats_path = resolve_path("_strategy_stats.json")
        if not isinstance(stats_path, Path):
            stats_path = resolve_path(stats_path)
        self.stats_path = stats_path
        self._state_lock = FileLock(str(self.state_path) + ".lock")
        self._stats_lock = FileLock(str(self.stats_path) + ".lock")
        self.keyword_map: Dict[str, str] = dict(keyword_map or KEYWORD_MAP)
        self.index: int = 0
        self.failure_counts: Dict[str, int] = {s: 0 for s in self.strategies}
        self.failure_limits: Dict[str, int] = {s: 1 for s in self.strategies}
        self.penalties: Dict[str, int] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._load_state()
        self._load_stats()

    # ------------------------------------------------------------------
    def _rotated(self) -> list[str]:
        if not self.strategies:
            return []
        return self.strategies[self.index:] + self.strategies[: self.index]

    # ------------------------------------------------------------------
    def select(self, selector: Callable[[Sequence[str]], str | None]) -> str | None:
        """Return the next strategy using ``selector`` for scoring."""

        ordered = self._rotated()
        if not ordered:
            return None
        return selector(ordered)

    # ------------------------------------------------------------------
    def ingest(
        self,
        strategy: str | None = None,
        failure_reason: str | None = None,
        roi_delta: float | None = None,
    ) -> str | None:
        """Ingest sandbox metrics for ``strategy``.

        Parameters
        ----------
        strategy:
            The strategy that was executed.  Defaults to the current strategy.
        failure_reason:
            Optional description of why the attempt failed.
        roi_delta:
            Change in ROI produced by the attempt.
        Returns
        -------
        str | None
            Strategy explicitly selected via keyword mapping, if any.
        """

        if not self.strategies:
            return None
        if strategy is None:
            strategy = self.strategies[self.index]
        roi = float(roi_delta or 0.0)
        success = failure_reason is None and roi >= 0
        if not success:
            self.failure_counts[strategy] = self.failure_counts.get(strategy, 0) + 1
        if failure_reason:
            reason_l = failure_reason.lower()
            for key, strat in self.keyword_map.items():
                if key in reason_l and strat in self.strategies:
                    self.index = self.strategies.index(strat)
                    self._save_state()
                    return strat
        self.update(strategy, roi, success)
        return None

    # ------------------------------------------------------------------
    def next(self) -> str | None:
        """Return the highest scoring strategy based on current stats."""

        if not self.strategies:
            return None
        pool = [
            s
            for s in self.strategies
            if self.failure_counts.get(s, 0) < self.failure_limits.get(s, 1)
        ]
        try:
            best = self.best_strategy(pool) if pool else None
        except Exception:  # pragma: no cover - guard against missing deps
            best = None
        if best:
            self.index = self.strategies.index(best)
        else:
            current = self.strategies[self.index]
            limit = self.failure_limits.get(current, 1)
            if self.failure_counts.get(current, 0) >= limit:
                self.index = (self.index + 1) % len(self.strategies)
        self._save_state()
        return self.strategies[self.index]

    # ------------------------------------------------------------------
    def record_failure(
        self,
        strategy: str | None = None,
        failure_reason: str | None = None,
        roi_delta: float | None = None,
    ) -> str | None:
        """Record a failed attempt and return the next strategy.

        Parameters
        ----------
        strategy:
            The strategy that was executed.
        failure_reason:
            Optional description of why the attempt failed.
        roi_delta:
            ROI change produced by the attempt. Defaults to ``-1.0`` when
            ``None``.
        """

        forced = self.ingest(
            strategy=strategy,
            failure_reason=failure_reason,
            roi_delta=-1.0 if roi_delta is None else roi_delta,
        )
        if forced:
            return forced
        return self.next()

    # ------------------------------------------------------------------
    def set_strategies(self, strategies: Sequence[str]) -> None:
        """Replace the managed strategies preserving rotation index."""

        self.strategies = list(strategies)
        if self.strategies:
            self.index %= len(self.strategies)
        else:
            self.index = 0
        # reset counts for new strategies
        self.failure_counts = {s: self.failure_counts.get(s, 0) for s in self.strategies}
        self.failure_limits = {s: self.failure_limits.get(s, 1) for s in self.strategies}
        self._save_state()

    # ------------------------------------------------------------------
    def update(self, strategy: str, roi: float, success: bool, weight: float = 1.0) -> None:
        """Update ROI statistics for ``strategy``."""

        rec = self.stats.setdefault(
            str(strategy),
            {
                "total": 0,
                "success": 0,
                "roi_sum": 0.0,
                "weighted_roi_sum": 0.0,
                "weight_sum": 0.0,
                "records": [],
            },
        )
        ts = time.time()
        rec["total"] += 1
        if success:
            rec["success"] += 1
            self.failure_counts[str(strategy)] = 0
        rec["roi_sum"] += float(roi)
        rec["weighted_roi_sum"] += float(roi) * float(weight)
        rec["weight_sum"] += float(weight)
        rec.setdefault("records", []).append(
            {"ts": ts, "roi": float(roi), "success": bool(success)}
        )
        mrec = self.metrics.setdefault(
            str(strategy), {"attempts": 0, "successes": 0, "roi": 0.0}
        )
        mrec["attempts"] += 1
        if success:
            mrec["successes"] += 1
        mrec["roi"] = float(roi)
        self._save_stats()
        self._save_state()

    # ------------------------------------------------------------------
    def best_strategy(self, strategies: Sequence[str]) -> str | None:
        """Return the strategy with the highest average ROI."""

        from menace_sandbox.sandbox_settings import SandboxSettings

        self._load_stats()
        self._load_state()
        penalties = self.penalties
        settings = SandboxSettings()
        threshold = settings.prompt_failure_threshold
        mult = settings.prompt_penalty_multiplier
        decay = getattr(settings, "prompt_roi_decay_rate", 0.0)
        now = time.time()
        eligible: list[tuple[str, float]] = []
        penalised: list[tuple[str, float]] = []
        for strat in strategies:
            count = penalties.get(str(strat), 0)
            rec = self.stats.get(str(strat), {})
            records = rec.get("records") or []
            score = 0.0
            if records:
                total_w = 0.0
                success_w = 0.0
                roi_w = 0.0
                for r in records:
                    ts = float(r.get("ts", now))
                    roi = float(r.get("roi", 0.0))
                    succ = bool(r.get("success"))
                    weight = math.exp(-decay * max(now - ts, 0.0)) if decay else 1.0
                    total_w += weight
                    roi_w += roi * weight
                    if succ:
                        success_w += weight
                if total_w:
                    success_rate = success_w / total_w
                    weighted_roi = roi_w / total_w
                    avg_weight = total_w / len(records)
                    score = success_rate * max(weighted_roi, 0.0) * avg_weight
            else:
                total = int(rec.get("total", 0))
                success = int(rec.get("success", 0))
                roi_sum = float(rec.get("roi_sum", 0.0))
                weighted_roi_sum = float(rec.get("weighted_roi_sum", 0.0))
                weight_sum = float(rec.get("weight_sum", 0.0))
                if total:
                    success_rate = success / total
                    if weight_sum:
                        weighted_roi = weighted_roi_sum / weight_sum
                    else:
                        weighted_roi = roi_sum / total
                    score = success_rate * max(weighted_roi, 0.0)
            score = score if score > 0 else 0.1
            weight = score * (mult if threshold and count >= threshold else 1.0)
            target = penalised if threshold and count >= threshold else eligible
            target.append((strat, weight))
        pool = eligible or penalised
        if not pool:
            return None
        return max(pool, key=lambda x: x[1])[0]

    # ------------------------------------------------------------------
    def load_penalties(self) -> Dict[str, int]:
        """Return a copy of stored penalty counts."""

        self._load_state()
        return dict(self.penalties)

    # ------------------------------------------------------------------
    def record_penalty(self, name: str) -> int:
        """Increment downgrade count for ``name`` and persist it."""
        self._load_state()
        self.penalties[name] = self.penalties.get(name, 0) + 1
        self._save_state()
        return self.penalties[name]

    # ------------------------------------------------------------------
    def reset_penalty(self, name: str) -> None:
        """Reset downgrade count for ``name`` to zero."""
        self._load_state()
        if name in self.penalties and self.penalties[name] != 0:
            self.penalties[name] = 0
            self._save_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        try:
            with self._state_lock:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
            stored = data.get("strategies")
            if stored:
                self.strategies = list(stored)
            self.index = int(data.get("index", 0))
            self.failure_counts.update({
                str(k): int(v) for k, v in data.get("failure_counts", {}).items()
            })
            self.failure_limits.update({
                str(k): int(v) for k, v in data.get("failure_limits", {}).items()
            })
            self.metrics.update(
                {
                    str(k): {
                        "attempts": int(v.get("attempts", 0)),
                        "successes": int(v.get("successes", 0)),
                        "roi": float(v.get("roi", 0.0)),
                    }
                    for k, v in data.get("metrics", {}).items()
                    if isinstance(v, dict)
                }
            )
            self.penalties.update({
                str(k): int(v) for k, v in data.get("penalties", {}).items()
            })
            if self.strategies:
                self.index %= len(self.strategies)
            else:
                self.index = 0
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "index": self.index,
                "strategies": self.strategies,
                "failure_counts": self.failure_counts,
                "failure_limits": self.failure_limits,
                "penalties": self.penalties,
                "metrics": self.metrics,
            }
            with self._state_lock:
                tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                os.replace(tmp, self.state_path)
        except Exception:  # pragma: no cover - best effort persistence
            pass

    # ------------------------------------------------------------------
    def _load_stats(self) -> None:
        try:
            with self._stats_lock:
                data = json.loads(self.stats_path.read_text(encoding="utf-8"))
            self.stats = {
                str(k): {
                    "total": int(v.get("total", 0)),
                    "success": int(v.get("success", 0)),
                    "roi_sum": float(v.get("roi_sum", 0.0)),
                    "weighted_roi_sum": float(v.get("weighted_roi_sum", 0.0)),
                    "weight_sum": float(v.get("weight_sum", 0.0)),
                    "records": [
                        {
                            "ts": float(r.get("ts", 0.0)),
                            "roi": float(r.get("roi", 0.0)),
                            "success": bool(r.get("success", False)),
                        }
                        for r in v.get("records", [])
                        if isinstance(r, dict)
                    ],
                }
                for k, v in data.items()
                if isinstance(v, dict)
            }
        except Exception:
            self.stats = {}

    # ------------------------------------------------------------------
    def _save_stats(self) -> None:
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self._stats_lock:
                tmp = self.stats_path.with_suffix(self.stats_path.suffix + ".tmp")
                tmp.write_text(json.dumps(self.stats), encoding="utf-8")
                os.replace(tmp, self.stats_path)
        except Exception:  # pragma: no cover - best effort persistence
            pass

    # ------------------------------------------------------------------
    @classmethod
    def load_strategy_stats(
        cls, path: str | Path | None = None
    ) -> Dict[str, Dict[str, float]]:
        p = Path(path) if path is not None else resolve_path("_strategy_stats.json")
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            return {}
        stats: Dict[str, Dict[str, float]] = {}
        for k, v in data.items():
            total = max(int(v.get("total", 0)), 1)
            success = int(v.get("success", 0))
            roi_sum = float(v.get("roi_sum", 0.0))
            weighted_roi_sum = float(v.get("weighted_roi_sum", 0.0))
            weight_sum = float(v.get("weight_sum", 0.0))
            success_rate = success / total
            if weight_sum:
                weighted_roi = weighted_roi_sum / weight_sum
            else:
                weighted_roi = roi_sum / total
            score = success_rate * max(weighted_roi, 0.0)
            stats[str(k)] = {
                "success_rate": success_rate,
                "weighted_roi": weighted_roi,
                "score": score,
            }
        return stats


__all__ = ["PromptStrategyManager"]
