import importlib
import sys
import types

import pytest


def test_warning_on_missing_sklearn(monkeypatch, caplog, tmp_path):
    caplog.set_level("WARNING")
    # stub out optional dependencies before import
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
    monkeypatch.setitem(sys.modules, "joblib", types.ModuleType("joblib"))
    cpb = importlib.import_module("menace.chatgpt_prediction_bot")
    bot = cpb.ChatGPTPredictionBot(tmp_path / "none.joblib")
    assert "scikit-learn" in caplog.text
    assert not cpb._SKLEARN_AVAILABLE
    assert isinstance(bot.pipeline, cpb.Pipeline)
    # cleanup to avoid side effects
    for m in list(sys.modules):
        if m.startswith("menace"):
            sys.modules.pop(m, None)


def test_fallback_prediction_stable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
    monkeypatch.setitem(sys.modules, "joblib", types.ModuleType("joblib"))
    cpb = importlib.import_module("menace.chatgpt_prediction_bot")

    bot = cpb.ChatGPTPredictionBot(
        tmp_path / "none.joblib", l2=0.1, iters=5, val_steps=1
    )
    idea = cpb.IdeaFeatures(
        market_type="tech",
        monetization_model="ads",
        startup_cost=1.0,
        skill=1.0,
        competition=2.0,
        uniqueness=3.0,
    )
    first = bot.predict(idea)
    second = bot.predict(idea)
    assert first == second
    assert 0.0 <= first[1] <= 1.0

    for m in list(sys.modules):
        if m.startswith("menace"):
            sys.modules.pop(m, None)
