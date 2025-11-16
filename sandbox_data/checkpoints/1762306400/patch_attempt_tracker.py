from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from .self_improvement.target_region import TargetRegion
except Exception:  # pragma: no cover - fallback when executed directly
    from self_improvement.target_region import TargetRegion  # type: ignore

try:  # pragma: no cover - optional dependency
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - fallback when executed directly
    import metrics_exporter as _me  # type: ignore


@dataclass
class _Keys:
    """Internal helper to build dictionary keys."""

    filename: str
    start_line: int
    end_line: int
    function: str

    @property
    def region(self) -> Tuple[str, int, int]:
        return (self.filename, self.start_line, self.end_line)

    @property
    def function_key(self) -> Tuple[str, str]:
        return (self.filename, self.function)


class PatchAttemptTracker:
    """Track patch attempts and handle escalation logic.

    Escalates from a single-line region to the containing function after two
    failures and to the entire module after two additional failures at the
    function level.  Escalation events are logged via the supplied logger which
    defaults to the :class:`AutoEscalationManager` logger.  If an
    ``escalation_counter`` gauge is provided, escalation events are also
    recorded in the ``patch_escalations_total`` metric labelled by escalation
    level.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        escalation_counter: _me.Gauge | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("AutoEscalationManager")
        self._region_failures: Dict[Tuple[str, int, int], int] = {}
        self._function_failures: Dict[Tuple[str, str], int] = {}
        self._escalation_counter = escalation_counter

    # ------------------------------------------------------------------
    def level_for(
        self, region: TargetRegion, func_region: TargetRegion
    ) -> tuple[str, TargetRegion | None]:
        """Return the patch level and region to operate on.

        After two failures at the function level the caller should rebuild the
        entire module.  A tuple of ("module", ``None``) is therefore returned
        to signal a full-module rewrite.
        """
        k = _Keys(region.filename, region.start_line, region.end_line, region.function)
        if self._function_failures.get(k.function_key, 0) >= 2:
            return "module", None
        elif self._region_failures.get(k.region, 0) >= 2:
            return "function", func_region
        return "region", region

    # ------------------------------------------------------------------
    def record_failure(self, level: str, region: TargetRegion, func_region: TargetRegion) -> None:
        """Record a failed attempt and log escalation events."""
        k = _Keys(region.filename, region.start_line, region.end_line, region.function)
        if level == "region":
            count = self._region_failures.get(k.region, 0) + 1
            self._region_failures[k.region] = count
            if count == 2:
                if self._escalation_counter is not None:
                    self._escalation_counter.labels(level="function").inc()
                self.logger.info(
                    "escalating patch region",
                    extra={
                        "level": "function",
                        "path": region.filename,
                        "function": region.function,
                    },
                )
        elif level == "function":
            count = self._function_failures.get(k.function_key, 0) + 1
            self._function_failures[k.function_key] = count
            if count == 2:
                if self._escalation_counter is not None:
                    self._escalation_counter.labels(level="module").inc()
                self.logger.info(
                    "escalating patch region",
                    extra={"level": "module", "path": func_region.filename},
                )

    # ------------------------------------------------------------------
    def attempts_for(self, region: TargetRegion) -> int:
        """Return total attempts for ``region`` across all escalation levels."""
        k = _Keys(region.filename, region.start_line, region.end_line, region.function)
        return (
            self._region_failures.get(k.region, 0)
            + self._function_failures.get(k.function_key, 0)
        )

    # ------------------------------------------------------------------
    def reset(self, region: TargetRegion) -> None:
        """Reset counters for ``region`` after a successful patch."""
        k = _Keys(region.filename, region.start_line, region.end_line, region.function)
        self._region_failures.pop(k.region, None)
        self._function_failures.pop(k.function_key, None)


__all__ = ["PatchAttemptTracker"]
