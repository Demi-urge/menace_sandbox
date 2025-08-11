import numpy as np
from menace_sandbox.adaptive_roi_predictor import AdaptiveROIPredictor


import pytest


@pytest.fixture()
def predictor(monkeypatch):
    """Return predictor with training dataset stubbed out."""

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.load_adaptive_roi_dataset",
        lambda: (np.empty((0, 2)), np.empty(0), np.empty(0)),
    )
    return AdaptiveROIPredictor()


def test_classifies_exponential_growth(predictor):
    feats = [[1, 0], [2, 1], [4, 2], [8, 4], [16, 8]]
    assert predictor.predict_growth_type(feats) == "exponential"


def test_classifies_linear_growth(predictor):
    feats = [[1, 0], [2, 1], [3, 1], [4, 1], [5, 1]]
    assert predictor.predict_growth_type(feats) == "linear"


def test_classifies_marginal_growth(predictor):
    feats = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    assert predictor.predict_growth_type(feats) == "marginal"
