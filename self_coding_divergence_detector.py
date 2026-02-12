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
    minimum_confidence: float = 1.0
    divergence_threshold_cycles: int = 2
    recovery_threshold_cycles: int = 2


@dataclass(frozen=True)
class DivergenceDetectionResult:
    triggered: bool
    reason: str | None
    reward_delta: float
    reward_trend: float
    real_metric_delta: float
    real_metric_trend: float
    metric_name: str | None
    confidence: float


def _trend_slope(values: Sequence[float]) -> float:
    """Compute a simple least-squares slope for equally spaced samples."""

    count = len(values)
    if count < 2:
        return 0.0
    x_mean = (count - 1) / 2.0
    y_mean = sum(values) / count
    numerator = 0.0
    denominator = 0.0
    for idx, value in enumerate(values):
        x_delta = idx - x_mean
        numerator += x_delta * (value - y_mean)
        denominator += x_delta * x_delta
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


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
        minimum_confidence=min(
            max(_safe_float(section.get("minimum_confidence"), defaults.minimum_confidence), 0.0),
            1.0,
        ),
        divergence_threshold_cycles=max(
            int(_safe_float(section.get("divergence_threshold_cycles"), defaults.divergence_threshold_cycles)),
            1,
        ),
        recovery_threshold_cycles=max(
            int(_safe_float(section.get("recovery_threshold_cycles"), defaults.recovery_threshold_cycles)),
            1,
        ),
    )


class SelfCodingDivergenceDetector:
    def __init__(self, config: DivergenceDetectorConfig) -> None:
        self.config = config

    def evaluate(self, records: Sequence[CycleMetricsRecord]) -> DivergenceDetectionResult:
        window = self.config.window_size
        if len(records) < window:
            return DivergenceDetectionResult(False, None, 0.0, 0.0, 0.0, 0.0, None, 0.0)

        sample = list(records)[-window:]
        reward_values = [item.reward_score for item in sample]
        reward_delta = sample[-1].reward_score - sample[0].reward_score
        reward_trend = _trend_slope(reward_values)
        if reward_delta < self.config.minimum_effect_size:
            return DivergenceDetectionResult(False, None, reward_delta, reward_trend, 0.0, 0.0, None, 0.0)

        if reward_trend <= 0.0:
            return DivergenceDetectionResult(False, None, reward_delta, reward_trend, 0.0, 0.0, None, 0.0)

        business_candidates: list[tuple[str, float, float, float]] = []
        for metric_name in ("profit", "revenue"):
            indexed_values: list[tuple[int, float]] = []
            for idx, row in enumerate(sample):
                raw = getattr(row, metric_name)
                if raw is None:
                    continue
                indexed_values.append((idx, float(raw)))
            confidence = len(indexed_values) / float(window)
            if len(indexed_values) < 2 or confidence < self.config.minimum_confidence:
                continue
            metric_delta = indexed_values[-1][1] - indexed_values[0][1]
            metric_trend = _trend_slope([value for _, value in indexed_values])
            business_candidates.append((metric_name, metric_delta, metric_trend, confidence))

        for metric_name, metric_delta, metric_trend, confidence in business_candidates:
            if metric_trend <= self.config.flatness_threshold:
                return DivergenceDetectionResult(
                    True,
                    "reward_trending_up_while_real_metric_down_or_flat",
                    reward_delta,
                    reward_trend,
                    metric_delta,
                    metric_trend,
                    metric_name,
                    confidence,
                )

        return DivergenceDetectionResult(False, None, reward_delta, reward_trend, 0.0, 0.0, None, 0.0)
