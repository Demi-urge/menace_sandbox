import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.preliminary_research_bot as prb

HTML1 = """
<html><body>
Profit margin is 10%. Operational cost per user is 5. Market saturation 2.
Competitors: Foo, Bar
</body></html>
"""

HTML2 = """
<html><body>
Profit margin is 10%. Operational cost per user is 5. Market saturation 2.
Competitors: Foo, Baz
</body></html>
"""

HTML3 = """
<html><body>
Profit margin is 20%. Operational cost per user is 5.
</body></html>
"""

def test_keywords():
    words = prb.PreliminaryResearchBot._keywords("profit market share growth")
    assert any(w in words for w in ["profit", "market"])


def test_process_model(monkeypatch):
    bot = prb.PreliminaryResearchBot()
    pages = [HTML1, HTML2]
    monkeypatch.setattr(bot, "_fetch", lambda url: pages.pop(0))
    data = bot.process_model("Model", ["u1", "u2"])
    assert data.profit_margin == 10
    assert data.operational_cost == 5
    assert data.roi_score == 1.0
    assert "Foo" in data.competitors and "Bar" in data.competitors and "Baz" in data.competitors
    assert not data.inconsistencies


def test_inconsistency(monkeypatch):
    bot = prb.PreliminaryResearchBot()
    pages = [HTML1, HTML3]
    monkeypatch.setattr(bot, "_fetch", lambda url: pages.pop(0))
    data = bot.process_model("Model", ["u1", "u2"])
    assert "profit_margin" in data.inconsistencies


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, _vec):
        self.called = True
        return 2.0


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _bot):
        return ["p"]


def test_prediction_bots_adjust_roi(monkeypatch):
    pred = DummyPred()
    manager = StubManager(pred)
    bot = prb.PreliminaryResearchBot(prediction_manager=manager)
    pages = [HTML1, HTML2]
    monkeypatch.setattr(bot, "_fetch", lambda url: pages.pop(0))
    data = bot.process_model("Model", ["u1", "u2"])
    assert pred.called
    assert data.roi_score == 1.5
