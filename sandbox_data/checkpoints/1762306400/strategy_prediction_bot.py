"""Strategy Prediction Bot for forecasting competitor moves and counter strategies."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import time
import logging

registry = BotRegistry()
data_bot = DataBot(start_server=False)

logger = logging.getLogger(__name__)

import numpy as np

if TYPE_CHECKING:
    from .prediction_manager_bot import PredictionManager
    from .niche_saturation_bot import NicheCandidate
    from .ai_counter_bot import TrafficSample
    from .sentiment_bot import FeedbackItem
    from .resource_prediction_bot import ResourceMetrics
if TYPE_CHECKING:
    from .research_aggregator_bot import ResearchAggregatorBot

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None  # type: ignore


@dataclass
class CompetitorFeatures:
    """Simple metrics about a competitor's current state."""

    revenue_growth: float
    funding: float
    sentiment: float
    tech_mentions: int

    def to_vector(self) -> List[float]:
        return [self.revenue_growth, self.funding, self.sentiment, float(self.tech_mentions)]


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class StrategyPredictionBot:
    """Predict competitor strategies and coordinate actions from other bots."""

    prediction_profile = {"scope": ["strategy"], "risk": ["medium"]}

    def __init__(
        self,
        prediction_manager: Optional["PredictionManager"] = None,
        *,
        aggregator: Optional["ResearchAggregatorBot"] = None,
    ) -> None:
        if LogisticRegression is not None:
            self.model = LogisticRegression()
        else:  # pragma: no cover - fallback simple model
            self.model = None
        self._weights: np.ndarray | None = None

        self.prediction_manager = prediction_manager
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.aggregator = aggregator

        self._niche_info: List["NicheCandidate"] = []
        self._ai_events: List["TrafficSample"] = []
        self._sentiment_events: List["FeedbackItem"] = []
        self._resource_events: List[Dict[str, "ResourceMetrics"]] = []
        self._efficiency_reports: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # data ingestion from other bots
    def receive_niche_info(self, niches: Iterable["NicheCandidate"]) -> None:
        self._niche_info.extend(list(niches))

    def receive_ai_competition(self, event: "TrafficSample") -> None:
        self._ai_events.append(event)

    def receive_sentiment(self, item: "FeedbackItem") -> None:
        self._sentiment_events.append(item)

    def receive_resource_usage(self, metrics: Dict[str, "ResourceMetrics"]) -> None:
        self._resource_events.append(metrics)

    def receive_efficiency_report(self, report: Dict[str, float]) -> None:
        """Store efficiency metrics provided by the efficiency bot."""
        self._efficiency_reports.append(report)

    # ------------------------------------------------------------------
    def formulate_strategy(self, topic: str = "strategy") -> str:
        """Create a simple textual strategy and pass to the aggregator."""
        parts: List[str] = []
        for n in self._niche_info:
            parts.append(f"Saturate {n.name} demand={n.demand}")
        for ev in self._ai_events:
            parts.append(f"Counter AI pattern {ev.pattern}")
        for s in self._sentiment_events:
            action = "boost" if s.sentiment > 0 else "mitigate"
            parts.append(f"{action} sentiment for {s.product}")
        for m in self._resource_events:
            for bot, r in m.items():
                parts.append(f"adjust {bot} cpu={r.cpu}")
        for rep in self._efficiency_reports:
            score = rep.get("predicted_bottleneck", 0.0)
            if score > 0.5:
                parts.append("mitigate bottleneck")
        strategy_text = "; ".join(parts)
        if self.aggregator:
            try:
                self.aggregator.requirements = [topic]
                self.aggregator.process(topic, energy=1)
            except Exception:
                logging.getLogger(__name__).exception("aggregator failed")
        return strategy_text

    def train(self, samples: Iterable[CompetitorFeatures], labels: Iterable[int]) -> None:
        """Train the underlying model."""
        X = np.array([s.to_vector() for s in samples])
        y = np.array(list(labels))
        if self.model is not None:
            self.model.fit(X, y)
            self._weights = None
        else:
            # fallback implementation of logistic regression using gradient descent
            Xb = np.hstack([np.ones((X.shape[0], 1)), X])
            weights = np.zeros(Xb.shape[1])
            lr = 0.1
            for _ in range(200):
                preds = 1.0 / (1.0 + np.exp(-Xb.dot(weights)))
                gradient = Xb.T.dot(preds - y) / Xb.shape[0]
                weights -= lr * gradient
            self._weights = weights

    def predict(self, features: CompetitorFeatures) -> float:
        """Return probability of aggressive strategic move."""
        if self.model is not None:
            prob = float(self.model.predict_proba([features.to_vector()])[0][1])
        elif self._weights is not None:
            vec = np.array([1.0, *features.to_vector()])
            score = float(vec.dot(self._weights))
            prob = float(1.0 / (1.0 + np.exp(-score)))
        else:
            # approximate logistic model when training data is unavailable
            score = (
                features.revenue_growth * 0.6
                + features.funding * 0.4
                + features.sentiment * 0.3
                + np.log1p(features.tech_mentions) * 0.2
            )
            prob = float(1.0 / (1.0 + np.exp(-score)))

        if self.prediction_manager:
            prob = self._apply_prediction_bots(prob, features)
        return float(prob)

    def counter_strategy(self, probability: float) -> str:
        """Formulate a simple counter strategy given predicted probability."""
        if probability >= 0.85:
            return "Initiate pre-emptive campaign and accelerate R&D"
        if probability >= 0.7:
            return "Launch aggressive marketing and monitor closely"
        if probability >= 0.55:
            return "Prepare defensive measures and adjust pricing"
        if probability >= 0.3:
            return "Monitor trends and strengthen partnerships"
        return "Maintain normal operations"

    def _apply_prediction_bots(
        self, base: float, features: CompetitorFeatures
    ) -> float:
        """Combine predictions from assigned prediction bots."""
        if not self.prediction_manager:
            return base
        prob = base
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            predict = getattr(entry.bot, "predict", None)
            if not callable(predict):
                continue
            try:
                other = float(predict(features.to_vector()))
                prob = (prob + other) / 2.0
            except Exception:
                continue
        return prob


_DISRUPTION_KEYWORDS = {"quantum", "gpt", "ai", "blockchain", "boom", "layoff", "pivot"}

try:
    from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CountVectorizer = None  # type: ignore

_vectorizer: CountVectorizer | None = None
_model: LogisticRegression | None = None

_TRAIN_DATA = [
    ("quantum breakthrough funding boom", 1),
    ("major ai pivot", 1),
    ("blockchain revolution", 1),
    ("massive layoffs announced", 1),
    ("routine marketing update", 0),
    ("minor feature release", 0),
    ("quarterly results stable", 0),
]

if LogisticRegression is not None and CountVectorizer is not None:
    _vectorizer = CountVectorizer()
    X = _vectorizer.fit_transform([t for t, _ in _TRAIN_DATA])
    y = [l for _, l in _TRAIN_DATA]
    _model = LogisticRegression(max_iter=200)
    try:
        _model.fit(X, y)
    except Exception:  # pragma: no cover - guard for datasets errors
        _model = None


def detect_disruption(signals: Iterable[str]) -> bool:
    """Return True if market disruption signals are found."""
    text = " ".join(signals).lower()
    if _model is not None and _vectorizer is not None:
        try:
            X = _vectorizer.transform([text])
            prob = float(_model.predict_proba(X)[0][1])
            return prob > 0.5
        except Exception:
            logging.getLogger(__name__).exception("disruption detect failed")
    hits = sum(1 for k in _DISRUPTION_KEYWORDS if k in text)
    ratio = hits / max(len(text.split()), 1)
    return ratio >= 0.1


__all__ = [
    "CompetitorFeatures",
    "StrategyPredictionBot",
    "detect_disruption",
]