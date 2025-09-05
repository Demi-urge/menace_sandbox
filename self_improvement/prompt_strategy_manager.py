"""Unified prompt strategy rotation and tracking utilities."""

from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
from typing import Callable, Dict, Sequence

from dynamic_path_router import resolve_path


DEFAULT_STRATEGIES: list[str] = [
    "strict_fix",
    "delete_rebuild",
    "comment_refactor",
    "unit_test_rewrite",
]

KEYWORD_MAP: Dict[str, str] = {
    "test": "unit_test_rewrite",
    "comment": "comment_refactor",
    "refactor": "comment_refactor",
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
        self.keyword_map: Dict[str, str] = dict(keyword_map or KEYWORD_MAP)
        self.index: int = 0
        self.failure_counts: Dict[str, int] = {s: 0 for s in self.strategies}
        self.failure_limits: Dict[str, int] = {s: 1 for s in self.strategies}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self._load_state()

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
    def record_failure(self, strategy: str | None = None, reason: str | None = None) -> str | None:
        """Record a failure and return the next strategy to try.

        Parameters
        ----------
        strategy:
            The strategy that failed.  Defaults to the current strategy.
        reason:
            Optional text describing the failure.  When a keyword from
            :attr:`keyword_map` is found the corresponding strategy is selected
            immediately.
        """

        if not self.strategies:
            return None
        if strategy is None:
            strategy = self.strategies[self.index]
        self.failure_counts[strategy] = self.failure_counts.get(strategy, 0) + 1
        if reason:
            reason_l = reason.lower()
            for key, strat in self.keyword_map.items():
                if key in reason_l and strat in self.strategies:
                    self.index = self.strategies.index(strat)
                    self._save_state()
                    return strat
        limit = self.failure_limits.get(strategy, 1)
        if self.failure_counts.get(strategy, 0) >= limit:
            try:
                idx = self.strategies.index(strategy)
            except ValueError:
                idx = self.index
            self.index = (idx + 1) % len(self.strategies)
        else:
            try:
                self.index = self.strategies.index(strategy)
            except ValueError:
                pass
        self._save_state()
        return self.strategies[self.index]

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
    def update(self, strategy: str, roi: float, success: bool) -> None:
        """Update ROI statistics for ``strategy``."""

        rec = self.metrics.setdefault(
            str(strategy), {"attempts": 0, "successes": 0, "roi": 0.0}
        )
        rec["attempts"] += 1
        if success:
            rec["successes"] += 1
            self.failure_counts[str(strategy)] = 0
        rec["roi"] += float(roi)
        self._save_state()

    # ------------------------------------------------------------------
    def best_strategy(self, strategies: Sequence[str]) -> str | None:
        """Return the strategy with the highest average ROI."""

        from . import prompt_memory
        from ..sandbox_settings import SandboxSettings

        penalties = prompt_memory.load_prompt_penalties()
        settings = SandboxSettings()
        threshold = settings.prompt_failure_threshold
        eligible: list[tuple[str, float]] = []
        penalised: list[tuple[str, float]] = []
        for strat in strategies:
            count = penalties.get(str(strat), 0)
            rec = self.metrics.get(str(strat), {})
            attempts = rec.get("attempts", 0)
            avg_roi = rec.get("roi", 0.0) / attempts if attempts else 0.0
            target = penalised if threshold and count >= threshold else eligible
            target.append((strat, avg_roi))
        pool = eligible or penalised
        if not pool:
            return None
        return max(pool, key=lambda x: x[1])[0]

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not self.strategies:
                self.strategies = list(data.get("strategies", []))
            self.index = int(data.get("index", 0))
            self.failure_counts.update({
                str(k): int(v) for k, v in data.get("failure_counts", {}).items()
            })
            self.failure_limits.update({
                str(k): int(v) for k, v in data.get("failure_limits", {}).items()
            })
            raw_metrics = data.get("metrics", {})
            self.metrics = {
                str(k): {
                    "attempts": int(v.get("attempts", 0)),
                    "successes": int(v.get("successes", 0)),
                    "roi": float(v.get("roi", 0.0)),
                }
                for k, v in raw_metrics.items()
                if isinstance(v, dict)
            }
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
                "metrics": self.metrics,
            }
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=self.state_path.parent, encoding="utf-8"
            ) as fh:
                json.dump(data, fh)
                tmp = Path(fh.name)
            os.replace(tmp, self.state_path)
        except Exception:  # pragma: no cover - best effort persistence
            pass


__all__ = ["PromptStrategyManager"]
