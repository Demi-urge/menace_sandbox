import numpy as np
from menace_sandbox.adaptive_roi_predictor import AdaptiveROIPredictor


import pytest


@pytest.fixture()
def predictor(monkeypatch):
    """Return predictor with training dataset stubbed out."""

    # Avoid training by supplying an empty dataset
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset",
        lambda: (np.empty((0, 2)), np.empty(0)),
    )
    return AdaptiveROIPredictor()


def test_predicts_exponential_growth(predictor):
    feats = [[1, 0], [2, 1], [4, 2], [8, 4], [16, 8]]
    roi, growth = predictor.predict(feats)
    assert growth == "exponential"
    assert roi == pytest.approx(16.0)


def test_predicts_linear_growth(predictor):
    feats = [[1, 0], [2, 1], [3, 1], [4, 1], [5, 1]]
    roi, growth = predictor.predict(feats)
    assert growth == "linear"
    assert roi == pytest.approx(5.0)


def test_predicts_marginal_growth(predictor):
    feats = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    roi, growth = predictor.predict(feats)
    assert growth == "marginal"
    assert roi == pytest.approx(1.0)
