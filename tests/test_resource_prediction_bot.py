import pytest

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
import sys
import types
from pathlib import Path

if "menace_sandbox" not in sys.modules:
    pkg = types.ModuleType("menace_sandbox")
    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["menace_sandbox"] = pkg

import menace_sandbox.resource_prediction_bot as rpb


@pytest.mark.skipif(pd is None, reason="pandas is required for historical DB tests")
def make_db(tmp_path):
    path = tmp_path / "hist.csv"
    df = pd.DataFrame([
        {"task": "t1", "cpu": 2.0, "memory": 100.0, "disk": 1.0, "time": 1.0},
        {"task": "t1", "cpu": 4.0, "memory": 120.0, "disk": 1.5, "time": 2.0},
        {"task": "t2", "cpu": 1.0, "memory": 80.0, "disk": 0.5, "time": 1.5},
    ])
    df.to_csv(path, index=False)
    return rpb.TemplateDB(path)


@pytest.mark.skipif(pd is None, reason="pandas is required for historical DB tests")
def test_predict(tmp_path):
    db = make_db(tmp_path)
    bot = rpb.ResourcePredictionBot(db)
    m = bot.predict("t1")
    assert abs(m.cpu - 3.0) < 1e-5
    assert m.disk > 1.0


def test_detect_redundancies():
    dups = rpb.ResourcePredictionBot.detect_redundancies(["a", "b", "a", "c", "b"])
    assert sorted(dups) == ["a", "b"]


@pytest.mark.skipif(pd is None, reason="pandas is required for historical DB tests")
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


def test_resource_prediction_without_pandas(monkeypatch, tmp_path):
    monkeypatch.setattr(rpb, "pd", None, raising=False)
    db = rpb.TemplateDB(tmp_path / "fallback.csv")
    bot = rpb.ResourcePredictionBot(db=db, data_bot=None, capital_bot=None)

    metrics = bot.predict("new-task")
    assert metrics.cpu == 1.0
    assert metrics.memory == 1.0
    assert metrics.disk == 10.0
    assert metrics.time == 1.0

    new_metrics = rpb.ResourceMetrics(cpu=2.0, memory=3.0, disk=4.0, time=5.0)
    db.add("new-task", new_metrics)
    db.save()

    updated = bot.predict("new-task")
    assert updated.cpu == pytest.approx(2.0)
    assert updated.memory == pytest.approx(3.0)
    assert updated.disk == pytest.approx(4.0)
    assert updated.time == pytest.approx(5.0)
