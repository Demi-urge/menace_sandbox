from __future__ import annotations

"""Utility to compute a numeric enhancement score.

The score is a weighted sum of several metrics that describe the effort and
quality of an enhancement.  Weights are loaded from
``config/enhancement_score.yaml`` so they can be tuned without code changes.
"""

from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML missing
    yaml = None  # type: ignore

from dynamic_path_router import resolve_path


@dataclass
class EnhancementMetrics:
    """Metrics describing a code enhancement."""

    lines_changed: int
    context_tokens: int
    time_to_completion: float
    tests_passed: int
    tests_failed: int
    error_traces: int
    effort_estimate: float


@dataclass
class EnhancementScoreWeights:
    """Weighting factors applied to :class:`EnhancementMetrics`."""

    difficulty: float = 1.0
    time_to_completion: float = 1.0
    tests_passed: float = 1.0
    tests_failed: float = -1.0
    error_traces: float = -1.0
    effort_estimate: float = 1.0


_DEFAULT_CONFIG_PATH = resolve_path("config/enhancement_score.yaml")


def load_weights(path: str | Path | None = None) -> EnhancementScoreWeights:
    """Load weighting factors from ``path``.

    ``path`` is resolved relative to the repository root when provided. If it
    is omitted, ``config/enhancement_score.yaml`` is used. Missing entries
    default to the values in :class:`EnhancementScoreWeights`.
    """

    cfg_path = resolve_path(path) if path else _DEFAULT_CONFIG_PATH
    data = {}
    if yaml is not None:
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            data = {}
    weights = EnhancementScoreWeights()
    for field in weights.__dataclass_fields__:  # type: ignore[attr-defined]
        if isinstance(data.get(field), (int, float)):
            setattr(weights, field, float(data[field]))
    return weights


def compute_enhancement_score(
    metrics: EnhancementMetrics, weights: EnhancementScoreWeights | None = None
) -> float:
    """Return a weighted enhancement score for ``metrics``.

    ``lines_changed`` and ``context_tokens`` are combined as a difficulty
    component before weighting. ``tests_failed`` and ``error_traces`` reduce the
    overall score when their weights are negative.
    """

    w = weights or load_weights()
    difficulty = metrics.lines_changed + metrics.context_tokens
    score = (
        difficulty * w.difficulty
        + metrics.time_to_completion * w.time_to_completion
        + metrics.tests_passed * w.tests_passed
        + metrics.tests_failed * w.tests_failed
        + metrics.error_traces * w.error_traces
        + metrics.effort_estimate * w.effort_estimate
    )
    return float(score)


__all__ = [
    "EnhancementMetrics",
    "EnhancementScoreWeights",
    "load_weights",
    "compute_enhancement_score",
]
