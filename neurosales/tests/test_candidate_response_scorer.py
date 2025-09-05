import os
import sys
import subprocess
import logging
import pytest
from unittest.mock import MagicMock
import types

from dynamic_path_router import resolve_path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util
import types

stub_sentiment = types.ModuleType("sentiment")
stub_sentiment.SentimentAnalyzer = lambda *a, **k: None
stub_user_prefs = types.ModuleType("user_preferences")
stub_user_prefs.PreferenceProfile = type("PP", (), {"embedding": []})

spec = importlib.util.spec_from_file_location(
    "neurosales.scoring",
    str(resolve_path("neurosales/neurosales/scoring.py")),
)
scoring = importlib.util.module_from_spec(spec)
sys.modules.setdefault("neurosales", types.ModuleType("neurosales"))
sys.modules["neurosales.sentiment"] = stub_sentiment
sys.modules["neurosales.user_preferences"] = stub_user_prefs
sys.modules["neurosales.scoring"] = scoring
spec.loader.exec_module(scoring)
sys.modules["neurosales"].scoring = scoring
CandidateResponseScorer = scoring.CandidateResponseScorer


def test_engagement_prediction_changes_after_training(tmp_path):
    model_path = resolve_path("neurosales") / "engagement_model.joblib"
    if model_path.exists():
        os.remove(model_path)

    scorer = CandidateResponseScorer()
    features = [30, 2, 1]
    baseline = scorer._predict_engagement(features)

    script = str(resolve_path("neurosales/scripts/train_engagement.py"))
    subprocess.check_call([sys.executable, script])

    trained = CandidateResponseScorer()
    new_pred = trained._predict_engagement(features)

    assert baseline != new_pred


def test_predict_engagement_logs_failure(caplog):
    scorer = CandidateResponseScorer()
    scorer._lr = MagicMock()
    scorer._lr.predict.side_effect = ValueError("bad")
    scorer._model_loaded = True
    scorer._np = type("np", (), {"array": lambda *a, **kw: [[1.0]]})

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            scorer._predict_engagement([1.0])
    assert any("Engagement prediction failed" in r.getMessage() for r in caplog.records)
