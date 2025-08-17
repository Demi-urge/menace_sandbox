from __future__ import annotations

"""Compute ROI scores from metric profiles with optional vetoes."""

from pathlib import Path
import logging
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "roi_profiles.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    _PROFILES: dict[str, dict[str, Any]] = yaml.safe_load(fh)


class ROICalculator:
    """Calculate ROI based on weighted metrics and veto rules."""

    def __init__(self, profile_type: str) -> None:
        try:
            profile = _PROFILES[profile_type]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"unknown ROI profile: {profile_type}") from exc
        self.profile_type = profile_type
        self.weights: dict[str, float] = profile.get("metrics", {})
        self.veto_exprs: list[str] = profile.get("veto", [])
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hard_fail = False

    # ------------------------------------------------------------------
    def _eval_vetoes(
        self, metrics: dict[str, float], flags: dict[str, Any]
    ) -> list[str]:
        context: dict[str, Any] = {**flags, **metrics, "true": True, "false": False}
        triggered: list[str] = []
        for expr in self.veto_exprs:
            expr_py = expr.replace(" true", " True").replace(" false", " False")
            try:
                if eval(expr_py, {"__builtins__": {}}, context):  # noqa: S307
                    triggered.append(expr)
            except Exception:  # pragma: no cover - debug logging only
                self.logger.exception("failed to evaluate veto expression %r", expr)
        return triggered

    # ------------------------------------------------------------------
    def calculate(
        self,
        metrics: dict[str, float],
        flags: dict[str, Any] | None = None,
    ) -> float:
        """Return ROI score for ``metrics`` or ``-inf`` if vetoes trigger."""
        flags = flags or {}
        triggered = self._eval_vetoes(metrics, flags)
        if triggered:
            self.hard_fail = True
            return float("-inf")
        self.hard_fail = False
        return sum(metrics.get(name, 0.0) * weight for name, weight in self.weights.items())

    # ------------------------------------------------------------------
    def log_debug(
        self,
        metrics: dict[str, float],
        flags: dict[str, Any] | None = None,
    ) -> None:
        """Log weights, per-metric contributions, final score, and vetoes."""
        flags = flags or {}
        contributions = {
            name: metrics.get(name, 0.0) * weight for name, weight in self.weights.items()
        }
        triggered = self._eval_vetoes(metrics, flags)
        score = float("-inf") if triggered else sum(contributions.values())
        self.logger.debug("weights: %s", self.weights)
        self.logger.debug("metrics: %s", metrics)
        self.logger.debug("flags: %s", flags)
        self.logger.debug("contributions: %s", contributions)
        self.logger.debug("final_score: %s", score)
        self.logger.debug("veto_triggered: %s", triggered)


__all__ = ["ROICalculator"]
