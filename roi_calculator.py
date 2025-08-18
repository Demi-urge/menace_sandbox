from __future__ import annotations

"""Profile-driven ROI scoring with configurable veto rules."""

from pathlib import Path
from typing import Any

import logging
import yaml


EXPECTED_METRICS = {
    "profitability",
    "efficiency",
    "reliability",
    "resilience",
    "maintainability",
    "security",
    "latency",
    "energy",
}


class ROICalculator:
    """Calculate ROI scores from weighted metrics and veto rules."""

    def __init__(
        self,
        profiles_path: str | Path = "configs/roi_profiles.yaml",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise calculator with profiles loaded from *profiles_path*.

        Parameters
        ----------
        profiles_path:
            Path to the YAML file containing ROI profiles.
        logger:
            Optional logger instance. If omitted, ``logging.getLogger(__name__)``
            is used.
        """
        path = Path(profiles_path)
        with path.open("r", encoding="utf-8") as fh:
            self.profiles: dict[str, dict[str, Any]] = yaml.safe_load(fh) or {}
        self._validate_profiles()
        self.hard_fail: bool = False
        self.logger = logger or logging.getLogger(__name__)

    def calculate(
        self, metrics: dict[str, Any], profile_type: str
    ) -> tuple[float, bool, list[str]]:
        """Return weighted ROI score and veto information.

        Missing metrics default to ``0.0``. When any veto condition is met the
        score becomes ``-inf`` and the veto list describes the triggered
        conditions.
        """
        try:
            profile = self.profiles[profile_type]
        except KeyError as exc:
            raise ValueError(f"unknown profile: {profile_type}") from exc

        weights: dict[str, float] = profile.get("weights", {})
        veto_rules: dict[str, dict[str, Any]] = profile.get("veto", {})

        score = 0.0
        for name, weight in weights.items():
            value = float(metrics.get(name, 0.0))
            score += value * weight

        triggers: list[str] = []
        for name, rule in veto_rules.items():
            value = metrics.get(name, 0.0 if any(k in rule for k in ("min", "max")) else None)
            if "min" in rule and float(value) < rule["min"]:
                triggers.append(f"{name} below min {rule['min']}")
            if "max" in rule and float(value) > rule["max"]:
                triggers.append(f"{name} above max {rule['max']}")
            if "equals" in rule and value == rule["equals"]:
                triggers.append(f"{name} equals {rule['equals']}")

        vetoed = bool(triggers)
        self.hard_fail = vetoed
        if vetoed:
            return (float("-inf"), True, triggers)
        return (score, False, [])

    def _validate_profiles(self) -> None:
        """Validate that profiles contain expected numeric weights."""
        for name, profile in self.profiles.items():
            weights = profile.get("weights", {})
            missing = EXPECTED_METRICS - weights.keys()
            if missing:
                raise ValueError(
                    f"profile '{name}' missing weight(s) for: {', '.join(sorted(missing))}"
                )
            non_numeric = [
                metric
                for metric, weight in weights.items()
                if isinstance(weight, bool)
                or not isinstance(weight, (int, float))
            ]
            if non_numeric:
                raise ValueError(
                    f"profile '{name}' has non-numeric weight(s) for: {', '.join(sorted(non_numeric))}"
                )
            total = float(
                sum(abs(float(weights[m])) for m in EXPECTED_METRICS)
            )
            if not (0.99 <= total <= 1.01):
                raise ValueError(
                    f"profile '{name}' weight absolute sum {total} outside acceptable range"
                )

    def log_debug(self, metrics: dict[str, Any], profile_type: str) -> None:
        """Log per-metric contributions, final score and veto triggers."""
        score, vetoed, triggers = self.calculate(metrics, profile_type)
        weights = self.profiles[profile_type].get("weights", {})
        for name, weight in weights.items():
            value = float(metrics.get(name, 0.0))
            self.logger.debug("%s * %s = %s", name, weight, value * weight)
        self.logger.debug("Final score: %s", score)
        if triggers:
            self.logger.debug("Veto triggers: %s", triggers)
        else:
            self.logger.debug("Veto triggers: none")

    def compute(self, metrics: dict[str, Any], profile_type: str) -> float:
        """Backward compatible alias for :meth:`calculate` returning only the score."""
        score, _, _ = self.calculate(metrics, profile_type)
        return score


__all__ = ["ROICalculator"]
