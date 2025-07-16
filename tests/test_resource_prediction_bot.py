import pytest

pytest.importorskip("pandas")

import pandas as pd
import menace.resource_prediction_bot as rpb


def make_db(tmp_path):
    path = tmp_path / "hist.csv"
    df = pd.DataFrame([
        {"task": "t1", "cpu": 2.0, "memory": 100.0, "disk": 1.0, "time": 1.0},
        {"task": "t1", "cpu": 4.0, "memory": 120.0, "disk": 1.5, "time": 2.0},
        {"task": "t2", "cpu": 1.0, "memory": 80.0, "disk": 0.5, "time": 1.5},
    ])
    df.to_csv(path, index=False)
    return rpb.TemplateDB(path)


def test_predict(tmp_path):
    db = make_db(tmp_path)
    bot = rpb.ResourcePredictionBot(db)
    m = bot.predict("t1")
    assert abs(m.cpu - 3.0) < 1e-5
    assert m.disk > 1.0


def test_detect_redundancies():
    dups = rpb.ResourcePredictionBot.detect_redundancies(["a", "b", "a", "c", "b"])
    assert sorted(dups) == ["a", "b"]


def test_optimise_schedule(tmp_path):
    db = make_db(tmp_path)
    bot = rpb.ResourcePredictionBot(db)
    tasks = ["t1", "t2"]
    order = bot.optimise_schedule(tasks, cpu_limit=3.5)
    assert set(order) == set(tasks)


def test_assess_risk(monkeypatch):
    metrics = rpb.ResourceMetrics(cpu=10.0, memory=50.0, disk=20.0, time=1.0)
    called = {}

    class FakeRisk:
        def risk(self, val):
            called["v"] = val
            return 0.5

    monkeypatch.setattr(rpb, "risky", FakeRisk())
    score = rpb.ResourcePredictionBot.assess_risk(metrics)
    assert called["v"] == metrics.time
    assert score == 0.5
