import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.pre_execution_roi_bot as prb


def _tasks():
    return [
        prb.BuildTask(
            name="a",
            complexity=1.0,
            frequency=1.0,
            expected_income=10.0,
            resources={"compute": 1.0, "storage": 1.0},
        ),
        prb.BuildTask(
            name="b",
            complexity=2.0,
            frequency=2.0,
            expected_income=20.0,
            resources={"compute": 2.0, "storage": 2.0},
        ),
    ]


def test_estimate_cost(tmp_path):
    db = prb.ROIHistoryDB(tmp_path / "hist.csv")
    db.add(1.0, 1.0, 0.1, 0.5, 5.0, 1.0)
    bot = prb.PreExecutionROIBot(db)
    cost = bot.estimate_cost(_tasks())
    assert cost > 0


def test_forecast_roi(tmp_path):
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"))
    result = bot.forecast_roi(_tasks(), projected_income=30.0)
    assert abs(result.roi - (result.income - result.cost)) < 1e-5


def test_run_scenario(tmp_path):
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"))
    base = bot.forecast_roi(_tasks(), 30.0)

    def remove_a(tasks):
        return [t for t in tasks if t.name != "a"]

    scenario = bot.run_scenario(_tasks(), remove_a)
    assert scenario.cost < base.cost


def test_diminishing_returns(tmp_path):
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"))
    res = bot.forecast_roi_diminishing(_tasks(), 1000.0)
    assert res.income <= bot.forecast_roi(_tasks(), 1000.0).income


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, feats):
        self.called = True
        return feats[0] + 1.0


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _):
        return ["p"]


def test_predict_model_roi_with_prediction(tmp_path):
    manager = StubManager(DummyPred())
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"), prediction_manager=manager)
    res = bot.predict_model_roi("m", _tasks())
    assert manager.registry["p"].bot.called
    assert res.roi != bot.forecast_roi(_tasks(), projected_income=30.0).roi


class DummyMetricPred:
    def __init__(self):
        self.args = None

    def predict_metric(self, name, feats):
        self.args = (name, feats)
        return 42.0


def test_predict_metric_helper(tmp_path):
    bot_obj = DummyMetricPred()
    manager = StubManager(bot_obj)
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"), prediction_manager=manager)
    val = bot.predict_metric("lucrativity", [1.0, 2.0])
    assert val == 42.0
    assert bot_obj.args[0] == "lucrativity"
