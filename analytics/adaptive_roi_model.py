from __future__ import annotations

"""Train and apply a GradientBoostingRegressor for ROI forecasting.

This module trains a model on data provided by
:mod:`analytics.adaptive_roi_dataset` and persists the fitted model under
``analytics/models``.  The resulting forecaster returns both a numerical ROI
forecast and a qualitative growth label:
``"exponential"`` for high growth, ``"linear"`` for moderate growth and
``"marginal"`` for low or negative growth.
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .adaptive_roi_dataset import build_dataset

# Path where the trained model is stored
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "adaptive_roi_gbr.joblib"


def train(save_path: Path | str = MODEL_PATH) -> GradientBoostingRegressor:
    """Train the ROI forecasting model and persist it.

    Parameters
    ----------
    save_path:
        Location where the trained model should be written.  Defaults to
        :data:`MODEL_PATH`.
    """
    data = build_dataset()
    if data.empty:
        raise RuntimeError("Training dataset is empty")

    features = data[["performance_delta", "gpt_score"]]
    target = data["roi_delta"]

    model = GradientBoostingRegressor(random_state=0)
    model.fit(features, target)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model


def load(model_path: Path | str = MODEL_PATH) -> GradientBoostingRegressor:
    """Load a previously trained model."""

    return joblib.load(model_path)


def classify_growth(roi: float) -> str:
    """Map a numerical ROI value to a qualitative growth label."""

    if roi > 0.75:
        return "exponential"
    if roi > 0.0:
        return "linear"
    return "marginal"


def forecast(
    model: GradientBoostingRegressor,
    performance_delta: float,
    gpt_score: float,
) -> Tuple[float, str]:
    """Return ROI forecast and growth label for a single data point."""

    df = pd.DataFrame(
        [[performance_delta, gpt_score]],
        columns=["performance_delta", "gpt_score"],
    )
    roi = float(model.predict(df)[0])
    return roi, classify_growth(roi)


__all__ = [
    "MODEL_DIR",
    "MODEL_PATH",
    "train",
    "load",
    "forecast",
    "classify_growth",
]
