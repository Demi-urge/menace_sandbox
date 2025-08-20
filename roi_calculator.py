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


REMEDIATION_HINTS: dict[str, str] = {
    "profitability": "optimise revenue streams; reduce costs",
    "efficiency": "optimise algorithms; reduce overhead",
    "reliability": "increase retries; improve test mocks",
    "resilience": "add fallbacks; improve monitoring",
    "maintainability": "refactor for clarity; improve documentation",
    "security": "harden authentication; add input validation",
    "latency": "optimise I/O; use caching",
    "energy": "batch work; reduce polling",
    "alignment_violation": "review alignment policies; add safeguards",
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


def propose_fix(
    metrics: dict[str, float], profile: dict[str, Any]
) -> list[tuple[str, str]]:
    """Return remediation hints for weakest metrics or veto violations.

    The profile's weights are used to determine each metric's contribution to the
    overall ROI score. The metrics with the lowest contributions are highlighted
    together with any metrics violating veto rules. The returned list contains up
    to three ``(metric, hint)`` pairs ordered by priority.
    """

    weights: dict[str, float] = profile.get("weights", {})
    veto_rules: dict[str, dict[str, Any]] = profile.get("veto", {})

    suggestions: list[tuple[str, str]] = []
    selected: set[str] = set()

    for name, rule in veto_rules.items():
        value = metrics.get(
            name, 0.0 if any(k in rule for k in ("min", "max")) else None
        )
        violated = False
        if "min" in rule and float(value) < rule["min"]:
            violated = True
        if "max" in rule and float(value) > rule["max"]:
            violated = True
        if "equals" in rule and value == rule["equals"]:
            violated = True
        if violated:
            suggestions.append((name, REMEDIATION_HINTS.get(name, f"improve {name}")))
            selected.add(name)

    contributions: list[tuple[str, float]] = []
    for name, weight in weights.items():
        value = float(metrics.get(name, 0.0))
        contributions.append((name, value * weight))

    contributions.sort(key=lambda item: item[1])

    for name, _ in contributions:
        if len(suggestions) >= 3:
            break
        if name in selected:
            continue
        suggestions.append((name, REMEDIATION_HINTS.get(name, f"improve {name}")))
        selected.add(name)

    return suggestions


__all__ = ["ROICalculator", "propose_fix"]
