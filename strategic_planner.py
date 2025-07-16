"""High level automated strategy planner."""

from __future__ import annotations

import logging

from .autoscaler import Autoscaler
from .strategy_prediction_bot import StrategyPredictionBot, CompetitorFeatures
from .trend_predictor import TrendPredictor


class StrategicPlanner:
    """Continuously refine objectives based on forecasts."""

    def __init__(
        self,
        strategy_bot: StrategyPredictionBot,
        autoscaler: Autoscaler,
        predictor: TrendPredictor,
    ) -> None:
        self.strategy_bot = strategy_bot
        self.autoscaler = autoscaler
        self.predictor = predictor
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def plan_cycle(self) -> str:
        """Compute a new strategy and adjust resources."""
        pred = self.predictor.predict_future_metrics(1)
        features = CompetitorFeatures(pred.roi, 1.0, 0.0, 1)
        prob = self.strategy_bot.predict(features)
        plan = self.strategy_bot.counter_strategy(prob)
        self.autoscaler.scale({"cpu": pred.roi})
        self.logger.info("strategy updated: %s", plan)
        return plan

