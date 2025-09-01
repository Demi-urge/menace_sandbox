from __future__ import annotations

"""Profile-driven ROI scoring with configurable veto rules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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


DEFAULT_REMEDIATION_HINTS: dict[str, str] = {
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


def _load_remediation_hints() -> dict[str, str]:
    """Return remediation hints loaded from ``configs/roi_fix_rules.yaml``.

    The YAML file can override or extend :data:`DEFAULT_REMEDIATION_HINTS`.
    Missing or invalid files fall back to the defaults.
    """

    try:
        with Path("configs/roi_fix_rules.yaml").open("r", encoding="utf-8") as fh:
            file_hints: dict[str, str] = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        file_hints = {}
    return {**DEFAULT_REMEDIATION_HINTS, **file_hints}


REMEDIATION_HINTS = _load_remediation_hints()


@dataclass
class ROIResult:
    """Structured result returned by :meth:`ROICalculator.calculate`.

    Attributes
    ----------
    score:
        Final ROI score.
    vetoed:
        ``True`` when any veto rule triggered.
    triggers:
        List of human readable veto descriptions.
    """

    score: float
    vetoed: bool
    triggers: list[str]

    def __iter__(self) -> Iterator[Any]:
        """Allow tuple unpacking for backward compatibility."""
        yield self.score
        yield self.vetoed
        yield self.triggers


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
        self.logger = logger or logging.getLogger(__name__)

    def calculate(self, metrics: dict[str, Any], profile_type: str) -> ROIResult:
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
        if vetoed:
            return ROIResult(float("-inf"), True, triggers)
        return ROIResult(score, False, [])

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
        result = self.calculate(metrics, profile_type)
        weights = self.profiles[profile_type].get("weights", {})
        for name, weight in weights.items():
            value = float(metrics.get(name, 0.0))
            self.logger.debug("%s * %s = %s", name, weight, value * weight)
        self.logger.debug("Final score: %s", result.score)
        if result.triggers:
            self.logger.debug("Veto triggers: %s", result.triggers)
        else:
            self.logger.debug("Veto triggers: none")

    def compute(self, metrics: dict[str, Any], profile_type: str) -> float:
        """Backward compatible alias for :meth:`calculate` returning only the score."""
        return self.calculate(metrics, profile_type).score


def propose_fix(
    metrics: dict[str, float], profile: str | dict[str, Any]
) -> list[tuple[str, str]]:
    """Return remediation hints for weakest metrics or veto violations.

    Remediation messages are loaded from ``configs/roi_fix_rules.yaml`` where
    they can be customised per metric. Results feed the ROI feedback loop by
    helping bots or humans prioritise follow-up actions.

    Parameters
    ----------
    metrics:
        Observed metric values.
    profile:
        Either a mapping containing ``weights`` and optional ``veto`` rules or
        the name of a profile to load from ``configs/roi_profiles.yaml``.

    The profile's weights are used to determine each metric's contribution to the
    overall ROI score. The metrics with the lowest contributions are highlighted
    together with any metrics violating veto rules. The returned list contains up
    to three ``(metric, hint)`` pairs ordered by priority. Unknown metrics fall
    back to a generic ``"improve <name>"`` suggestion. If the profile name cannot
    be resolved the function returns an empty list.
    """

    if isinstance(profile, str):
        try:
            with Path("configs/roi_profiles.yaml").open(
                "r", encoding="utf-8"
            ) as fh:
                profiles: dict[str, Any] = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            profiles = {}
        profile_dict: dict[str, Any] = profiles.get(profile, {})
    else:
        profile_dict = profile or {}

    weights: dict[str, float] = profile_dict.get("weights", {})
    veto_rules: dict[str, dict[str, Any]] = profile_dict.get("veto", {})

    suggestions: list[tuple[str, str]] = []
    selected: set[str] = set()

    def _add(metric: str) -> None:
        if metric in selected or len(suggestions) >= 3:
            return
        hint = REMEDIATION_HINTS.get(metric, f"improve {metric}")
        suggestions.append((metric, hint))
        selected.add(metric)

    for name, rule in veto_rules.items():
        value = metrics.get(
            name, 0.0 if any(k in rule for k in ("min", "max")) else None
        )
        if (
            ("min" in rule and float(value) < rule["min"])
            or ("max" in rule and float(value) > rule["max"])
            or ("equals" in rule and value == rule["equals"])
        ):
            _add(name)

    contributions = sorted(
        (
            (name, float(metrics.get(name, 0.0)) * weight)
            for name, weight in weights.items()
        ),
        key=lambda item: item[1],
    )

    for name, _ in contributions:
        _add(name)
        if len(suggestions) >= 3:
            break

    return suggestions


__all__ = ["ROICalculator", "ROIResult", "propose_fix"]
