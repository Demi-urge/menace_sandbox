"""Reusable metric generation helpers for tests.

These utilities provide deterministic metrics for bots so integration
and unit tests can simulate ROI and error changes without touching the
real metrics pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class StaticMetricGenerator:
    """Produce fixed ROI and error metrics for a bot.

    Parameters
    ----------
    roi:
        Return on investment value.
    errors:
        Error count associated with the metrics.
    """

    roi: float
    errors: float = 0.0

    def generate(self) -> Dict[str, float]:
        """Return the metrics as a dictionary for ``DataBot`` helpers."""
        return {"roi": self.roi, "errors": self.errors}
