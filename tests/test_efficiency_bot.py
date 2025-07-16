import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.efficiency_bot as eb
import menace.data_bot as db


def test_compress_and_log(tmp_path):
    edb = eb.EfficiencyDB(tmp_path / "e.db")
    bot = eb.EfficiencyBot(edb)
    models = {"m1": 50.0, "m2": 30.0}
    res = bot.optimise(models)
    hist = edb.history()
    assert len(res) == 2
    assert len(hist) == 2
    assert hist[0][0] == "m1"
    assert hist[0][1] <= 0.5


def test_compress_smaller():
    assert eb.EfficiencyBot.compress_model(20.0) <= 20.0


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, vec):
        self.called = True
        return 0.8


class DummyManager:
    def __init__(self, bot):
        self.registry = {"x": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _bot):
        return ["x"]


class DummyStrategy:
    def __init__(self):
        self.reports = []

    def receive_efficiency_report(self, rep):
        self.reports.append(rep)


def test_efficiency_reporting(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    data = db.DataBot(mdb)
    data.collect("b", response_time=0.2)
    manager = DummyManager(DummyPred())
    strat = DummyStrategy()
    bot = eb.EfficiencyBot(eb.EfficiencyDB(tmp_path / "e.db"), data_bot=data, prediction_manager=manager, strategy_bot=strat)
    report = bot.assess_efficiency()
    bot.send_findings(report)
    assert "predicted_bottleneck" in report
    assert strat.reports
    assert manager.registry["x"].bot.called
