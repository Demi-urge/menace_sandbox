"""Predict future ROI change from evolution history."""

from __future__ import annotations

from .bot_registry import BotRegistry

from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass
from typing import List

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .evolution_predictor import EvolutionPredictor

from .evolution_history_db import EvolutionHistoryDB

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:  # optional dependency
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover - sklearn missing
    LinearRegression = None  # type: ignore


@dataclass
class PredictedROI:
    """Prediction result for an action."""

    action: str
    expected_roi: float


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class EvolutionAnalysisBot:
    """Train a simple model on ``EvolutionHistoryDB`` records."""

    def __init__(
        self,
        history_db: EvolutionHistoryDB | None = None,
        data_bot: DataBot | None = None,
        capital_bot: CapitalManagementBot | None = None,
    ) -> None:
        self.history = history_db or EvolutionHistoryDB()
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.model = LinearRegression() if LinearRegression else None
        self.actions: List[str] = []

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Fit the regression model from historical events."""
        rows = self.history.fetch(200)
        if not rows or self.model is None:
            return
        action_map: dict[str, int] = {}
        X: list[list[float]] = []
        y: list[float] = []
        engagement = (
            self.data_bot.engagement_delta(limit=50) if self.data_bot else 0.0
        )
        roi_trend = (
            self.data_bot.long_term_roi_trend(limit=200) if self.data_bot else 0.0
        )
        profit_trend = (
            self.capital_bot.profit_trend() if self.capital_bot else 0.0
        )
        volatility = anomaly_ratio = 0.0
        if self.data_bot:
            try:
                df = self.data_bot.db.fetch(200)
                if hasattr(df, "empty"):
                    if not getattr(df, "empty", True):
                        df["roi"] = df["revenue"] - df["expense"]
                        volatility = float(df["roi"].std() or 0.0)
                        anomaly_ratio = float(
                            len(DataBot.detect_anomalies(df, "roi")) / len(df)
                        )
                elif isinstance(df, list) and df:
                    rois = [float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in df]
                    mean = sum(rois) / len(rois)
                    volatility = float(
                        (sum((r - mean) ** 2 for r in rois) / len(rois)) ** 0.5
                    )
                    df_list = [{"roi": r} for r in rois]
                    anomaly_ratio = float(
                        len(DataBot.detect_anomalies(df_list, "roi")) / len(rois)
                    )
            except Exception:
                volatility = anomaly_ratio = 0.0
        worst = self.data_bot.worst_bot(limit=200) if self.data_bot else None
        for row in rows:
            action = row[0]
            before_m = row[1]
            roi = row[3]
            eff = row[5]
            bottleneck = row[6]
            if action not in action_map:
                action_map[action] = len(action_map)
            complexity = 0.0
            if self.data_bot:
                try:
                    complexity = float(
                        DataBot.complexity_score(self.data_bot.db.fetch(50))
                    )
                except Exception:
                    complexity = 0.0
            bot_roi = 0.0
            if self.capital_bot:
                try:
                    bot_roi = float(self.capital_bot.bot_roi(action))
                except Exception:
                    bot_roi = 0.0
            X.append(
                [
                    float(action_map[action]),
                    float(before_m),
                    float(engagement),
                    float(profit_trend),
                    float(roi_trend),
                    float(volatility),
                    float(anomaly_ratio),
                    float(complexity),
                    float(bot_roi),
                    float(eff),
                    float(bottleneck),
                    float(1.0 if action == worst else 0.0),
                ]
            )
            y.append(float(roi))
        self.actions = [
            a for a, _ in sorted(action_map.items(), key=lambda p: p[1])
        ]
        try:
            self.model.fit(X, y)
        except Exception:
            self.model = None

    def predict(self, action: str, before_metric: float) -> float:
        """Return the expected ROI for ``action`` given ``before_metric``."""
        if not self.model or action not in self.actions:
            return 0.0
        idx = self.actions.index(action)
        engagement = (
            self.data_bot.engagement_delta(limit=50) if self.data_bot else 0.0
        )
        roi_trend = (
            self.data_bot.long_term_roi_trend(limit=200) if self.data_bot else 0.0
        )
        profit_trend = (
            self.capital_bot.profit_trend() if self.capital_bot else 0.0
        )
        volatility = anomaly_ratio = 0.0
        if self.data_bot:
            try:
                df = self.data_bot.db.fetch(200)
                if hasattr(df, "empty"):
                    if not getattr(df, "empty", True):
                        df["roi"] = df["revenue"] - df["expense"]
                        volatility = float(df["roi"].std() or 0.0)
                        anomaly_ratio = float(
                            len(DataBot.detect_anomalies(df, "roi")) / len(df)
                        )
                elif isinstance(df, list) and df:
                    rois = [float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in df]
                    mean = sum(rois) / len(rois)
                    volatility = float(
                        (sum((r - mean) ** 2 for r in rois) / len(rois)) ** 0.5
                    )
                    df_list = [{"roi": r} for r in rois]
                    anomaly_ratio = float(
                        len(DataBot.detect_anomalies(df_list, "roi")) / len(rois)
                    )
            except Exception:
                volatility = anomaly_ratio = 0.0
        complexity = 0.0
        if self.data_bot:
            try:
                complexity = float(
                    DataBot.complexity_score(self.data_bot.db.fetch(50))
                )
            except Exception:
                complexity = 0.0
        bot_roi = 0.0
        if self.capital_bot:
            try:
                bot_roi = float(self.capital_bot.bot_roi(action))
            except Exception:
                bot_roi = 0.0
        worst = self.data_bot.worst_bot(limit=200) if self.data_bot else None
        eff = bottleneck = 0.0
        if self.data_bot:
            try:
                df = self.data_bot.db.fetch(30)
                if hasattr(df, "empty"):
                    if not getattr(df, "empty", True):
                        eff = float(max(0.0, 100.0 - df["cpu"].mean()))
                        if "errors" in df.columns:
                            bottleneck = float(df["errors"].mean())
                elif isinstance(df, list) and df:
                    avg_cpu = sum(r.get("cpu", 0.0) for r in df) / len(df)
                    eff = float(max(0.0, 100.0 - avg_cpu))
                    bottleneck = float(
                        sum(r.get("errors", 0.0) for r in df) / len(df)
                    )
            except Exception:
                eff = bottleneck = 0.0
        try:
            return float(
                self.model.predict(
                    [
                        [
                            float(idx),
                            float(before_metric),
                            float(engagement),
                            float(profit_trend),
                            float(roi_trend),
                            float(volatility),
                            float(anomaly_ratio),
                            float(complexity),
                            float(bot_roi),
                            float(eff),
                            float(bottleneck),
                            float(1.0 if action == worst else 0.0),
                        ]
                    ]
                )[0]
            )
        except Exception:
            return 0.0


__all__ = ["PredictedROI", "EvolutionAnalysisBot", "EvolutionPredictor"]