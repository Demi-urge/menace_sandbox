from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import logging

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml may be unavailable in slim envs
    yaml = None  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - flat layout fallback
    from dynamic_path_router import resolve_path  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleMetricsRecord:
    """Cycle-level metrics schema used to evaluate divergence."""

    cycle_index: int
    bot_id: str
    workflow_id: str
    reward_score: float
    revenue: float | None = None
    profit: float | None = None


@dataclass(frozen=True)
class DivergenceDetectorConfig:
    window_size: int = 3
    flatness_threshold: float = 0.0
    minimum_effect_size: float = 0.1


@dataclass(frozen=True)
class DivergenceDetectionResult:
    triggered: bool
    reason: str | None
    reward_delta: float
    real_metric_delta: float
    metric_name: str | None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_divergence_detector_config(config_path: str | None = None) -> DivergenceDetectorConfig:
    """Load detector thresholds from config and return safe defaults on failure."""

    path = Path(config_path or resolve_path("config/self_coding_divergence_guard.yaml"))
    defaults = DivergenceDetectorConfig()
    if not path.exists() or yaml is None:
        return defaults
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.exception("failed to parse divergence config: %s", path)
        return defaults
    if not isinstance(payload, Mapping):
        return defaults
    section = payload.get("self_coding_divergence_guard", payload)
    if not isinstance(section, Mapping):
        section = {}
    return DivergenceDetectorConfig(
        window_size=max(int(_safe_float(section.get("window_size"), defaults.window_size)), 2),
        flatness_threshold=_safe_float(section.get("flatness_threshold"), defaults.flatness_threshold),
        minimum_effect_size=max(
            _safe_float(section.get("minimum_effect_size"), defaults.minimum_effect_size), 0.0
        ),
    )


class SelfCodingDivergenceDetector:
    def __init__(self, config: DivergenceDetectorConfig) -> None:
        self.config = config

    def evaluate(self, records: Sequence[CycleMetricsRecord]) -> DivergenceDetectionResult:
        window = self.config.window_size
        if len(records) < window:
            return DivergenceDetectionResult(False, None, 0.0, 0.0, None)

        sample = list(records)[-window:]
        reward_delta = sample[-1].reward_score - sample[0].reward_score
        if reward_delta < self.config.minimum_effect_size:
            return DivergenceDetectionResult(False, None, reward_delta, 0.0, None)

        business_candidates: list[tuple[str, float]] = []
        if all(r.profit is not None for r in sample):
            profit_delta = float(sample[-1].profit) - float(sample[0].profit)
            business_candidates.append(("profit", profit_delta))
        if all(r.revenue is not None for r in sample):
            revenue_delta = float(sample[-1].revenue) - float(sample[0].revenue)
            business_candidates.append(("revenue", revenue_delta))

        for metric_name, metric_delta in business_candidates:
            if metric_delta <= self.config.flatness_threshold:
                return DivergenceDetectionResult(
                    True,
                    "reward_trending_up_while_real_metric_down_or_flat",
                    reward_delta,
                    metric_delta,
                    metric_name,
                )

        return DivergenceDetectionResult(False, None, reward_delta, 0.0, None)
