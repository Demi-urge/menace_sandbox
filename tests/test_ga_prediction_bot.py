import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np
import menace.ga_prediction_bot as gp
import pandas as pd


def test_ga_prediction(tmp_path):
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, size=20)
    db = gp.TemplateDB(tmp_path / "templates.csv")
    bot = gp.GAPredictionBot(X, y, pop_size=4, db=db)
    rec = bot.evolve(generations=1)
    assert isinstance(rec.score, float)
    assert bot.evaluation_count() > 0
    assert len(db.df) >= 1


def test_ga_prediction_with_manager(tmp_path):
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=10)

    class DummyDB:
        def fetch(self, limit: int = 20):
            return pd.DataFrame({"cpu": [10.0], "errors": [0]})

    class DummyData:
        def __init__(self):
            self.db = DummyDB()

        def collect(self, **kw):
            pass

    class DummyCapital:
        def update_rois(self):
            pass

        def energy_score(self, **kw):
            return 0.5

    data_bot = DummyData()
    capital = DummyCapital()

    class DummyManager:
        def __init__(self):
            self.registry = {}

        def register_bot(self, bot, profile):
            self.registry["stub"] = type(
                "Entry",
                (),
                {"id": "stub", "bot": bot, "profile": profile},
            )
            return "stub"

        def assign_prediction_bots(self, _):
            return list(self.registry.keys())

    manager = DummyManager()

    class StubPred:
        def __init__(self):
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            return [0.0 for _ in X]

    stub = StubPred()
    bid = manager.register_bot(stub, {"scope": ["prediction"]})
    tdb = gp.TemplateDB(tmp_path / "t.csv")
    bot = gp.GAPredictionBot(
        X,
        y,
        pop_size=3,
        db=tdb,
        data_bot=data_bot,
        capital_bot=capital,
        prediction_manager=manager,
    )
    assert bid in bot.assigned_prediction_bots
    bot.evolve(generations=1)
    assert stub.calls > 0

