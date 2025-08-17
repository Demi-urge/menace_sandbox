from __future__ import annotations

"""Profile-driven ROI scoring with configurable veto rules."""

from pathlib import Path
from typing import Any

import yaml


class ROICalculator:
    """Calculate ROI scores from weighted metrics and veto rules."""

    def __init__(self, profiles_path: str | Path = "configs/roi_profiles.yaml") -> None:
        """Initialise calculator with profiles loaded from *profiles_path*."""
        path = Path(profiles_path)
        with path.open("r", encoding="utf-8") as fh:
            self.profiles: dict[str, dict[str, Any]] = yaml.safe_load(fh) or {}
        self.hard_fail: bool = False

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

    def log_debug(self, metrics: dict[str, Any], profile_type: str) -> None:
        """Print per-metric contributions, final score and veto triggers."""
        score, vetoed, triggers = self.calculate(metrics, profile_type)
        weights = self.profiles[profile_type].get("weights", {})
        for name, weight in weights.items():
            value = float(metrics.get(name, 0.0))
            print(f"{name} * {weight} = {value * weight}")
        print(f"Final score: {score}")
        if triggers:
            print(f"Veto triggers: {triggers}")
        else:
            print("Veto triggers: none")

    def compute(self, metrics: dict[str, Any], profile_type: str) -> float:
        """Backward compatible alias for :meth:`calculate` returning only the score."""
        score, _, _ = self.calculate(metrics, profile_type)
        return score


__all__ = ["ROICalculator"]
