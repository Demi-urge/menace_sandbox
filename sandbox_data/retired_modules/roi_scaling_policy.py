from __future__ import annotations

"""Autoscaling decisions based on ROI forecasts."""

import logging
from typing import Iterable

from .resource_prediction_bot import ResourcePredictionBot
from .autoscaler import Autoscaler


class ROIScalingPolicy:
    """Select scaling actions using ROI predictions."""

    def __init__(
        self,
        predictor: ResourcePredictionBot | None = None,
        autoscaler: Autoscaler | None = None,
        *,
        roi_threshold: float = 0.1,
    ) -> None:
        self.predictor = predictor or ResourcePredictionBot()
        self.autoscaler = autoscaler or Autoscaler()
        self.roi_threshold = roi_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def evaluate_and_scale(self, horizon: int = 24) -> None:
        try:
            forecast = self.predictor.predict_roi(horizon=horizon)
        except Exception:
            self.logger.exception("forecast failed")
            return
        roi = float(forecast.get("roi", 0.0))
        if roi > self.roi_threshold:
            self.logger.info("scaling up based on ROI %.3f", roi)
            self.autoscaler.scale_up()
        elif roi < -self.roi_threshold:
            self.logger.info("scaling down based on ROI %.3f", roi)
            self.autoscaler.scale_down()


__all__ = ["ROIScalingPolicy"]
