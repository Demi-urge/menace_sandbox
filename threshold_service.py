from __future__ import annotations

"""Service encapsulating threshold management and notifications."""

from pathlib import Path
from typing import Dict, Optional
import logging

try:  # pragma: no cover - optional type import
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - flat layout fallback
    from unified_event_bus import UnifiedEventBus  # type: ignore

from .roi_thresholds import ROIThresholds
from .sandbox_settings import SandboxSettings
from .self_coding_thresholds import (
    get_thresholds,
    update_thresholds,
    SelfCodingThresholds,
    _load_config as _load_threshold_config,
    flush_deferred_threshold_writes,
)

try:  # pragma: no cover - allow flat imports
    from .shared_event_bus import event_bus as _SHARED_EVENT_BUS
except Exception:  # pragma: no cover - flat layout fallback
    from shared_event_bus import event_bus as _SHARED_EVENT_BUS  # type: ignore


logger = logging.getLogger(__name__)


class ThresholdService:
    """Load, cache and persist self-coding thresholds."""

    def __init__(self, event_bus: Optional[UnifiedEventBus] = None) -> None:
        self.event_bus = event_bus or _SHARED_EVENT_BUS
        self._thresholds: Dict[str, ROIThresholds] = {}

    # ------------------------------------------------------------------
    def _publish(self, bot: str, rt: ROIThresholds) -> None:
        if not self.event_bus:
            return
        payload = {
            "bot": bot,
            "roi_drop": rt.roi_drop,
            "error_threshold": rt.error_threshold,
            "test_failure_threshold": rt.test_failure_threshold,
            "patch_success_drop": rt.patch_success_drop,
        }
        try:  # pragma: no cover - best effort
            self.event_bus.publish("thresholds:updated", payload)
        except Exception:
            logger.exception("failed to publish threshold update")

    # ------------------------------------------------------------------
    def load(
        self,
        bot: str | None = None,
        settings: SandboxSettings | None = None,
        *,
        bootstrap_mode: bool | None = None,
    ) -> SelfCodingThresholds:
        """Return raw thresholds for *bot* from configuration."""
        return get_thresholds(bot, settings, bootstrap_fast=bool(bootstrap_mode))

    # ------------------------------------------------------------------
    def reload(
        self,
        bot: str | None = None,
        settings: SandboxSettings | None = None,
        *,
        bootstrap_mode: bool | None = None,
    ) -> ROIThresholds:
        """Refresh cached thresholds for *bot* and emit update when changed."""
        raw = self.load(bot, settings, bootstrap_mode=bootstrap_mode)
        rt = ROIThresholds(
            roi_drop=raw.roi_drop,
            error_threshold=raw.error_increase,
            test_failure_threshold=raw.test_failure_increase,
            patch_success_drop=raw.patch_success_drop,
        )
        key = bot or ""
        prev = self._thresholds.get(key)
        self._thresholds[key] = rt
        if bot:
            data = _load_threshold_config()
            bots = data.get("bots", {}) if isinstance(data, dict) else {}
            if bot not in bots:
                update_thresholds(
                    bot,
                    roi_drop=rt.roi_drop,
                    error_increase=rt.error_threshold,
                    test_failure_increase=rt.test_failure_threshold,
                    patch_success_drop=rt.patch_success_drop,
                )
        if bot and prev != rt:
            self._publish(bot, rt)
        return rt

    # ------------------------------------------------------------------
    def get(
        self,
        bot: str | None = None,
        settings: SandboxSettings | None = None,
        *,
        bootstrap_mode: bool | None = None,
    ) -> ROIThresholds:
        """Return cached thresholds for *bot* loading them if missing."""
        key = bot or ""
        if key not in self._thresholds:
            return self.reload(bot, settings, bootstrap_mode=bootstrap_mode)
        return self._thresholds[key]

    # ------------------------------------------------------------------
    def update(
        self,
        bot: str,
        *,
        roi_drop: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        patch_success_drop: float | None = None,
        bootstrap_mode: bool | None = None,
    ) -> None:
        """Persist new thresholds for *bot* and broadcast changes."""
        bootstrap_mode = bool(bootstrap_mode)
        update_thresholds(
            bot,
            roi_drop=roi_drop,
            error_increase=error_threshold,
            test_failure_increase=test_failure_threshold,
            patch_success_drop=patch_success_drop,
            bootstrap_safe=bootstrap_mode,
        )
        if bootstrap_mode:
            logger.info(
                "threshold update for %s deferred during bootstrap; cached for later flush",
                bot,
            )
        current = self._thresholds.get(bot)
        if current is None:
            current = self.reload(bot)
        new_rt = ROIThresholds(
            roi_drop=roi_drop if roi_drop is not None else current.roi_drop,
            error_threshold=(
                error_threshold if error_threshold is not None else current.error_threshold
            ),
            test_failure_threshold=(
                test_failure_threshold
                if test_failure_threshold is not None
                else current.test_failure_threshold
            ),
            patch_success_drop=(
                patch_success_drop
                if patch_success_drop is not None
                else current.patch_success_drop
            ),
        )
        prev = self._thresholds.get(bot)
        self._thresholds[bot] = new_rt
        if prev != new_rt:
            self._publish(bot, new_rt)

    # ------------------------------------------------------------------
    def flush_bootstrap_writes(self, path: Path | None = None) -> bool:
        """Write any deferred bootstrap threshold updates to disk."""

        flushed = flush_deferred_threshold_writes(path)
        if flushed:
            logger.info(
                "flushed deferred threshold updates to %s", path or "config/self_coding_thresholds.yaml"
            )
        else:
            logger.debug("no deferred threshold updates to flush")
        return flushed


# Shared service instance used across the project
threshold_service = ThresholdService()

__all__ = ["ThresholdService", "threshold_service"]
