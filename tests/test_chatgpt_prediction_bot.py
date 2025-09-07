import importlib
import sys
import types
from pathlib import Path

import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

joblib = pytest.importorskip("joblib")

pkg = types.ModuleType("menace")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
pkg.RAISE_ERRORS = False
sys.modules["menace"] = pkg

vector_service_pkg = types.ModuleType("vector_service")
vector_service_pkg.__path__ = []
vector_service_pkg.SharedVectorService = object
vector_service_pkg.CognitionLayer = object


class _StubContextBuilder:
    def refresh_db_weights(self):
        pass


ctx_mod = types.ModuleType("vector_service.context_builder")
ctx_mod.ContextBuilder = _StubContextBuilder
sys.modules["vector_service"] = vector_service_pkg
sys.modules["vector_service.context_builder"] = ctx_mod
sys.modules["menace.shared_gpt_memory"] = types.SimpleNamespace(GPT_MEMORY_MANAGER=None)


cpb = importlib.import_module("menace.chatgpt_prediction_bot")


def build_model(path: Path) -> None:
    X = [
        {
            "market_type": "tech",
            "monetization_model": "ads",
            "startup_cost": 1.0,
            "skill": 1.0,
            "competition": 3.0,
            "uniqueness": 2.0,
        },
        {
            "market_type": "finance",
            "monetization_model": "subscription",
            "startup_cost": 5.0,
            "skill": 2.0,
            "competition": 1.0,
            "uniqueness": 4.0,
        },
    ]
    y = [0, 1]
    vec = DictVectorizer(sparse=False)
    clf = LogisticRegression()
    pipe = Pipeline([("vec", vec), ("clf", clf)])
    pipe.fit(X, y)
    joblib.dump(pipe, path)


def test_prediction(tmp_path):
    model_path = tmp_path / "model.joblib"
    build_model(model_path)
    bot = cpb.ChatGPTPredictionBot(model_path)
    idea = cpb.IdeaFeatures(
        market_type="finance",
        monetization_model="subscription",
        startup_cost=4.0,
        skill=3.0,
        competition=0.5,
        uniqueness=3.0,
    )
    result, score = bot.predict(idea)
    assert isinstance(result, bool)
    assert 0.0 <= score <= 1.0


def test_batch_prediction(tmp_path):
    model_path = tmp_path / "model.joblib"
    build_model(model_path)
    bot = cpb.ChatGPTPredictionBot(model_path)
    ideas = [
        cpb.IdeaFeatures(
            market_type="tech",
            monetization_model="ads",
            startup_cost=1.0,
            skill=1.0,
            competition=3.0,
            uniqueness=2.0,
        ),
        cpb.IdeaFeatures(
            market_type="finance",
            monetization_model="subscription",
            startup_cost=5.0,
            skill=2.0,
            competition=1.0,
            uniqueness=4.0,
        ),
    ]
    results = bot.batch_predict(ideas)
    assert len(results) == 2


def test_evaluate_enhancement():
    bot = cpb.ChatGPTPredictionBot.__new__(cpb.ChatGPTPredictionBot)
    bot.pipeline = None  # bypass loading
    ev = bot.evaluate_enhancement("Improve", "Adds more features and efficiency")
    assert -1.0 <= ev.value <= 1.0
    assert ev.description and ev.reason
