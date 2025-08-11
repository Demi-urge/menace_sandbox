from __future__ import annotations

"""Predict ROI growth patterns and ROI estimates using a lightweight model.

This module trains a simple regression model on the aggregated ROI dataset
returned by :func:`adaptive_roi_dataset.build_dataset`.  Given a sequence of
improvement feature vectors it forecasts ROI over the provided horizon and
classifies the projected curve as ``"exponential"``, ``"linear"`` or
``"marginal"``.  Trained model parameters are stored on disk to allow
incremental retraining when new data becomes available.
"""

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
except Exception:  # pragma: no cover - sklearn missing
    GradientBoostingRegressor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LinearRegression  # type: ignore
except Exception:  # pragma: no cover - sklearn missing
    LinearRegression = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover - joblib missing
    joblib = None  # type: ignore

import pickle

from .adaptive_roi_dataset import build_dataset

__all__ = ["AdaptiveROIPredictor", "predict_growth_type", "predict"]


class AdaptiveROIPredictor:
    """Train a lightweight model to forecast ROI and growth patterns.

    The model is trained on the dataset produced by :func:`build_dataset` and
    persisted to ``model_path`` so that future instances can reuse the learned
    parameters without retraining from scratch.
    """

    def __init__(self, model_path: str | Path = "adaptive_roi_model.pkl") -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._load()
        if self._model is None:
            self.train()

    # ------------------------------------------------------------------
    # persistence helpers
    def _load(self) -> None:
        """Load model parameters from ``self.model_path`` if available."""

        if self.model_path.exists():
            try:
                if joblib is not None:
                    self._model = joblib.load(self.model_path)
                else:
                    with self.model_path.open("rb") as fh:
                        self._model = pickle.load(fh)
            except Exception:  # pragma: no cover - corrupted file
                self._model = None

    def _save(self) -> None:
        """Persist the current model to disk."""

        if self._model is None:
            return
        try:  # pragma: no cover - disk issues
            if joblib is not None:
                joblib.dump(self._model, self.model_path)
            else:
                with self.model_path.open("wb") as fh:
                    pickle.dump(self._model, fh)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # training
    def train(self, dataset: Tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """Fit the underlying model on ``dataset``.

        Parameters
        ----------
        dataset:
            Optional tuple ``(features, targets)``.  When omitted the data is
            loaded using :func:`build_dataset`.
        """

        if dataset is None:
            X, y = build_dataset()
        else:
            X, y = dataset
        if X.size == 0 or y.size == 0:
            self._model = None
            return
        if GradientBoostingRegressor is not None:
            model = GradientBoostingRegressor(random_state=0)
        elif LinearRegression is not None:
            model = LinearRegression()
        else:  # pragma: no cover - extremely minimal environment
            model = None
        self._model = model
        if self._model is not None:
            try:
                self._model.fit(X, y)
                self._save()
            except Exception:  # pragma: no cover - training failure
                self._model = None

    # ------------------------------------------------------------------
    def _predict_sequence(self, features: np.ndarray) -> np.ndarray:
        """Return model predictions for ``features``.

        ``features`` must be a ``(n_samples, n_features)`` array.  When the
        model is unavailable the first column is returned unchanged as a naive
        baseline.
        """

        if features.ndim != 2:
            raise ValueError("features must be 2D")
        if getattr(self._model, "predict", None) is not None:
            try:
                return np.asarray(self._model.predict(features), dtype=float)
            except Exception:  # pragma: no cover - prediction failure
                pass
        return features[:, 0].astype(float)

    def _classify_growth(self, preds: np.ndarray) -> str:
        """Classify growth pattern of ``preds`` as exponential, linear or marginal."""

        if preds.size == 0:
            return "marginal"
        times = np.arange(len(preds), dtype=float)

        # linear fit
        lin_coef = np.polyfit(times, preds, 1)
        lin_pred = np.polyval(lin_coef, times)
        lin_err = float(np.mean((preds - lin_pred) ** 2))

        # exponential fit (via log transform)
        offset = 1 - preds.min() if np.any(preds <= 0) else 0.0
        log_vals = np.log(preds + offset)
        exp_coef = np.polyfit(times, log_vals, 1)
        exp_pred = np.exp(np.polyval(exp_coef, times)) - offset
        exp_err = float(np.mean((preds - exp_pred) ** 2))

        slope = lin_coef[0]
        if exp_err < lin_err * 0.8 and slope > 0:
            return "exponential"
        if abs(slope) < 0.01:
            return "marginal"
        return "linear"

    # ------------------------------------------------------------------
    # public API
    def predict(self, improvement_features: Sequence[Sequence[float]]) -> tuple[float, str]:
        """Return ``(roi_estimate, growth_category)`` for the provided features."""

        feats = np.asarray(list(improvement_features), dtype=float)
        preds = self._predict_sequence(feats)
        roi_estimate = float(preds[-1]) if preds.size else 0.0
        growth = self._classify_growth(preds)
        return roi_estimate, growth

    # Backwards compatible wrapper
    def predict_growth_type(self, action_features: Sequence[Sequence[float]]) -> str:
        """Return only the growth classification for ``action_features``."""

        return self.predict(action_features)[1]


# Module‑level convenience instance -----------------------------------------
_predictor: AdaptiveROIPredictor | None = None


def predict_growth_type(action_features: Sequence[Sequence[float]]) -> str:
    """Return growth classification for ``action_features`` using a singleton."""

    global _predictor
    if _predictor is None:
        _predictor = AdaptiveROIPredictor()
    return _predictor.predict_growth_type(action_features)


def predict(action_features: Sequence[Sequence[float]]) -> tuple[float, str]:
    """Return ``(roi_estimate, growth_category)`` using a module-level predictor."""

    global _predictor
    if _predictor is None:
        _predictor = AdaptiveROIPredictor()
    return _predictor.predict(action_features)

