"""Shared configuration for ROI and error thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from .sandbox_settings import SandboxSettings


@dataclass(frozen=True)
class ROIThresholds:
    """Container for ROI and error thresholds."""

    roi_drop: float
    error_threshold: float


def load_thresholds(settings: SandboxSettings | None = None) -> ROIThresholds:
    """Return threshold configuration using :class:`SandboxSettings`.

    Parameters
    ----------
    settings:
        Optional settings instance; if omitted a default ``SandboxSettings`` will
        be created.
    """

    s = settings or SandboxSettings()
    return ROIThresholds(roi_drop=s.self_coding_roi_drop, error_threshold=s.self_coding_error_increase)
