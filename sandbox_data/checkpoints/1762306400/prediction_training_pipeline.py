from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import logging

from .prediction_manager_bot import PredictionManager
from .data_bot import DataBot, MetricsDB
from .capital_management_bot import CapitalManagementBot

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class TrainingOutcome:
    """Result for a single prediction bot training cycle."""

    bot_id: str
    accuracy: float
    generations: int


class PredictionTrainingPipeline:
    """Coordinate evaluation and evolution of prediction bots."""

    def __init__(
        self,
        manager: PredictionManager | None = None,
        data_bot: DataBot | None = None,
        capital_bot: CapitalManagementBot | None = None,
        *,
        threshold: float = 0.8,
        max_generations: int = 3,
    ) -> None:
        self.manager = manager or PredictionManager()
        self.data_bot = data_bot or DataBot(MetricsDB(), capital_bot=capital_bot)
        self.capital_bot = capital_bot or CapitalManagementBot(data_bot=self.data_bot)
        if self.manager.data_bot is None:
            self.manager.data_bot = self.data_bot
        if self.manager.capital_bot is None:
            self.manager.capital_bot = self.capital_bot
        self.threshold = threshold
        self.max_generations = max_generations

    # ------------------------------------------------------------------
    def _accuracy(self, bot_id: str) -> float:
        df = self.data_bot.db.fetch(50)
        if pd is None:
            rows = [r for r in df if r.get("bot") == bot_id]
            if not rows:
                base = 1.0
            else:
                err = sum(float(r.get("errors", 0.0)) for r in rows) / len(rows)
                base = 1.0 - err / (err + 1.0)
        else:
            df = df[df["bot"] == bot_id]
            if df.empty:
                base = 1.0
            else:
                err = float(df["errors"].mean() or 0.0)
                base = 1.0 - err / (err + 1.0)
        if self.capital_bot:
            try:
                roi = self.capital_bot.bot_roi(bot_id)
                base = (base + min(1.0, roi / 100.0)) / 2.0
            except Exception as exc:
                logger.warning("failed to fetch ROI for %s: %s", bot_id, exc)
        return base

    def _roi_forecast(self, limit: int = 20) -> float:
        """ROI forecast using linear regression or ARIMA if available."""
        if pd is None:
            return 0.0
        try:
            df = self.data_bot.db.fetch(limit)
            if df.empty:
                return 0.0
            df["roi"] = df["revenue"] - df["expense"]
            df = df.reset_index(drop=True)
            if len(df) < 2:
                return float(df["roi"].iloc[-1])
            try:
                from statsmodels.tsa.arima.model import ARIMA

                model = ARIMA(df["roi"], order=(1, 1, 1)).fit()
                pred = model.forecast()[0]
                return float(pred)
            except Exception as exc:
                logger.exception(
                    "ARIMA ROI forecast failed; falling back to LinearRegression",
                    exc,
                )
                from sklearn.linear_model import LinearRegression

                X = df.index.values.reshape(-1, 1)
                y = df["roi"].values
                model = LinearRegression().fit(X, y)
                pred = model.predict([[len(df)]])[0]
                return float(pred)
        except Exception as exc:
            logger.exception("_roi_forecast failure: %s", exc)
            return 0.0

    def train(self, bot_ids: Iterable[str]) -> List[TrainingOutcome]:
        results: List[TrainingOutcome] = []
        for bid in bot_ids:
            entry = self.manager.registry.get(bid)
            if entry is None:
                continue
            accuracy = self._accuracy(bid)
            forecast = self._roi_forecast()
            generations = 0
            while accuracy < self.threshold and generations < self.max_generations:
                generations += 1
                new_entries = self.manager.trigger_evolution(entry.profile)
                new_entry = new_entries[0]
                if hasattr(new_entry.bot, "evolve"):
                    try:
                        new_entry.bot.evolve(generations=1)
                    except Exception as exc:
                        logger.warning("bot evolution failed for %s: %s", new_entry.id, exc)
                self.data_bot.collect(bot=new_entry.id, response_time=0.0, errors=0)
                if self.capital_bot:
                    try:
                        self.capital_bot.update_rois()
                    except Exception as exc:
                        logger.warning("update_rois failed: %s", exc)
                accuracy = self._accuracy(new_entry.id)
                self.manager.retire_bot(bid)
                bid = new_entry.id
                entry = new_entry
            results.append(TrainingOutcome(bot_id=bid, accuracy=accuracy, generations=generations))
            if self.data_bot:
                try:
                    self.data_bot.db.log_eval("training_pipeline", "roi_forecast", forecast)
                except Exception as exc:
                    logger.warning("failed to log roi_forecast: %s", exc)
        return results


__all__ = ["TrainingOutcome", "PredictionTrainingPipeline"]
