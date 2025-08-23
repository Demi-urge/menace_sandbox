import pytest

pytest.importorskip("numpy")

import menace.ai_counter_bot as acb
from menace.db_router import init_db_router
from unittest.mock import patch


def test_detect_ai_presence():
    assert acb.detect_ai_presence("using GPT automation")
    assert not acb.detect_ai_presence("human generated content")


def test_counter_db_roundtrip(tmp_path):
    router = init_db_router("t", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        db = acb.CounterDB(router=router)
        db.add("text", True, "rule-based", "counter flood")
        rows = db.fetch()
        gc.assert_called_with("events")
    assert rows and rows[0][1] is True


def test_prediction_training():
    bot = acb.AICounterBot()
    samples = [
        acb.TrafficSample("p", 5, 0.1, 0.2),
        acb.TrafficSample("p", 40, 0.2, 0.8),
    ]
    labels = [0, 1]
    bot.train_predictor(samples, labels)
    prob = bot.predict_adaptation(samples[1])
    assert 0 <= prob <= 1


def test_bot_analyse(tmp_path):
    router = init_db_router("u", str(tmp_path / "l.db"), str(tmp_path / "s.db"))
    db = acb.CounterDB(router=router)
    bot = acb.AICounterBot(db)
    samples = [
        acb.TrafficSample("p", 1, 0.1, 0.1),
        acb.TrafficSample("p", 5, 0.2, 0.8),
    ]
    bot.train_predictor(samples, [0, 1])
    detected, alg, counter = bot.analyse(["This post uses GPT automation"])
    assert detected
    rows = db.fetch()
    assert rows and rows[0][2] == alg


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, _vec):
        self.called = True
        return 1.0


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _bot):
        return ["p"]


class DummyStrategy:
    def __init__(self):
        self.events = []

    def receive_ai_competition(self, event):
        self.events.append(event)


def test_integration_with_strategy(tmp_path):
    manager = StubManager(DummyPred())
    strategy = DummyStrategy()
    router = init_db_router("i", str(tmp_path / "l2.db"), str(tmp_path / "s2.db"))
    bot = acb.AICounterBot(
        db=acb.CounterDB(router=router),
        prediction_manager=manager,
        strategy_bot=strategy,
    )
    sample = acb.TrafficSample("pat", 10, 0.1, 0.2)
    bot.train_predictor([sample, acb.TrafficSample("pat", 30, 0.2, 0.8)], [0, 1])
    prob = bot.predict_adaptation(sample)
    assert prob > 0.5
    detected, alg, _ = bot.analyse(["using GPT here"])
    assert detected
    assert strategy.events and strategy.events[0].pattern == alg
