import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.strategy_prediction_bot as spb
from dataclasses import dataclass, field


@dataclass
class NicheCandidate:
    name: str
    demand: float
    competition: float
    trend: float = 0.0


@dataclass
class TrafficSample:
    pattern: str
    frequency: int
    timing_std: float
    similarity: float


@dataclass
class FeedbackItem:
    text: str
    product: str
    source: str
    sentiment: float = 0.0


@dataclass
class ResourceMetrics:
    cpu: float
    memory: float
    disk: float
    time: float


def test_training_and_prediction():
    samples = [
        spb.CompetitorFeatures(0.1, 1.0, 0.5, 2),
        spb.CompetitorFeatures(0.5, 2.0, 0.8, 5),
        spb.CompetitorFeatures(-0.2, 0.5, -0.3, 1),
        spb.CompetitorFeatures(0.3, 1.5, 0.2, 3),
    ]
    labels = [0, 1, 0, 1]
    bot = spb.StrategyPredictionBot()
    bot.train(samples, labels)
    prob = bot.predict(spb.CompetitorFeatures(0.4, 1.8, 0.6, 4))
    assert 0 <= prob <= 1


def test_counter_strategy():
    bot = spb.StrategyPredictionBot()
    assert "aggressive" in bot.counter_strategy(0.8).lower()
    assert "defensive" in bot.counter_strategy(0.5).lower()
    assert "normal" in bot.counter_strategy(0.1).lower()


def test_detect_disruption():
    assert spb.detect_disruption(["Quantum breakthrough", "funding boom"]) is True
    assert not spb.detect_disruption(["routine update", "marketing"])


class DummyPred:
    def predict(self, vec):
        return 0.9


class DummyAgg:
    def __init__(self):
        self.called = False

    def process(self, topic, energy=1):
        self.called = True
        return []

    class InfoDB:
        def add(self, item):
            pass

    @property
    def info_db(self):
        return self.InfoDB()


class StubManager:
    def __init__(self, bot):
        self.registry = {"x": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _bot):
        return ["x"]


def test_strategy_integration(tmp_path):
    manager = StubManager(DummyPred())
    agg = DummyAgg()
    bot = spb.StrategyPredictionBot(prediction_manager=manager, aggregator=agg)
    bot.receive_niche_info([NicheCandidate("ai", 1.0, 0.1, 0.2)])
    bot.receive_ai_competition(TrafficSample("pat", 10, 0.5, 0.8))
    bot.receive_sentiment(FeedbackItem(text="good", product="m", source="x", sentiment=0.5))
    bot.receive_resource_usage({"b": ResourceMetrics(cpu=1.0, memory=1.0, disk=1.0, time=1.0)})
    strategy = bot.formulate_strategy()
    assert "ai" in strategy
    assert agg.called

