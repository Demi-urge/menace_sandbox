"""Shared configuration for ROI and error thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from .sandbox_settings import SandboxSettings
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sandbox_settings import BotThresholds


@dataclass(frozen=True)
class ROIThresholds:
    """Container for ROI and error thresholds."""

    roi_drop: float
    error_threshold: float
    test_failure_threshold: float = 0.0
    patch_success_drop: float = -0.2


def load_thresholds(
    bot: str | None = None, settings: SandboxSettings | None = None
) -> ROIThresholds:
    """Return threshold configuration using :class:`SandboxSettings`.

    Parameters
    ----------
    bot:
        Optional bot name to lookup per-bot thresholds.
    settings:
        Optional settings instance; if omitted a default ``SandboxSettings`` will
        be created.
    """

    s = settings or SandboxSettings()
    roi_drop = s.self_coding_roi_drop
    err_thresh = s.self_coding_error_increase
    fail_thresh = getattr(s, "self_coding_test_failure_increase", 0.0)
    patch_drop = getattr(s, "self_coding_patch_success_drop", -0.2)
    if bot and bot in s.bot_thresholds:
        bt: BotThresholds = s.bot_thresholds[bot]
        if bt.roi_drop is not None:
            roi_drop = bt.roi_drop
        if bt.error_threshold is not None:
            err_thresh = bt.error_threshold
        if getattr(bt, "test_failure_threshold", None) is not None:
            fail_thresh = bt.test_failure_threshold
        if getattr(bt, "patch_success_drop", None) is not None:
            patch_drop = bt.patch_success_drop
    return ROIThresholds(
        roi_drop=roi_drop,
        error_threshold=err_thresh,
        test_failure_threshold=fail_thresh,
        patch_success_drop=patch_drop,
    )
