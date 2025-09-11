"""Utilities for self-coding ROI/error thresholds.

The configuration file ``config/self_coding_thresholds.yaml`` stores
thresholds keyed by bot name.  Each entry may define ``roi_drop``
(allowable decrease in ROI) and ``error_increase`` (maximum allowed
increase in errors).  This module provides helpers to load and update
these thresholds so that different components can share the same
trigger logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

from .sandbox_settings import SandboxSettings
from .dynamic_path_router import resolve_path


@dataclass(frozen=True)
class SelfCodingThresholds:
    """Container for self-coding trigger thresholds."""

    roi_drop: float
    error_increase: float


_CONFIG_PATH = resolve_path("config/self_coding_thresholds.yaml")


def _load_config(path: Path | None = None) -> Dict[str, dict]:
    try:
        return yaml.safe_load((path or _CONFIG_PATH).read_text()) or {}
    except Exception:
        return {}


def get_thresholds(
    bot: str | None = None,
    settings: SandboxSettings | None = None,
    *,
    path: Path | None = None,
) -> SelfCodingThresholds:
    """Return thresholds for ``bot``.

    Parameters
    ----------
    bot:
        Bot name to lookup.  When omitted the default thresholds are
        returned.
    settings:
        Optional ``SandboxSettings`` providing baseline values when the
        configuration file is missing.
    path:
        Optional override for the configuration path.
    """

    s = settings or SandboxSettings()
    roi_drop = getattr(s, "self_coding_roi_drop", -0.1)
    err_inc = getattr(s, "self_coding_error_increase", 1.0)
    data = _load_config(path)
    default = data.get("default", {})
    bots = data.get("bots", {})
    roi_drop = float(default.get("roi_drop", roi_drop))
    err_inc = float(default.get("error_increase", err_inc))
    if bot and bot in bots:
        cfg = bots[bot] or {}
        roi_drop = float(cfg.get("roi_drop", roi_drop))
        err_inc = float(cfg.get("error_increase", err_inc))
    return SelfCodingThresholds(roi_drop=roi_drop, error_increase=err_inc)


def update_thresholds(
    bot: str,
    *,
    roi_drop: float | None = None,
    error_increase: float | None = None,
    path: Path | None = None,
) -> None:
    """Persist new thresholds for ``bot`` to the configuration file."""

    cfg_path = path or _CONFIG_PATH
    data = _load_config(cfg_path)
    bots = data.setdefault("bots", {})
    cfg = bots.setdefault(bot, {})
    if roi_drop is not None:
        cfg["roi_drop"] = float(roi_drop)
    if error_increase is not None:
        cfg["error_increase"] = float(error_increase)
    try:
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:
        # best effort – calling code may log failures
        pass


__all__ = ["SelfCodingThresholds", "get_thresholds", "update_thresholds"]
