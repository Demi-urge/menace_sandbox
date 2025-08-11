from __future__ import annotations

"""Predict ROI growth patterns using a lightweight model.

This module trains a simple regression model on the aggregated ROI dataset
returned by :func:`adaptive_roi_dataset.load_adaptive_roi_dataset`.  Given a
sequence of action feature vectors it forecasts the corresponding ROI values
and categorises the trajectory as ``"exponential"``, ``"linear"`` or
``"marginal"``.

The implementation intentionally keeps dependencies minimal.  When
``sklearn`` is available a :class:`~sklearn.ensemble.GradientBoostingRegressor`
provides the forecasts.  Otherwise a basic least‑squares linear model is
used.
"""

from typing import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
except Exception:  # pragma: no cover - fallback when sklearn missing
    GradientBoostingRegressor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LinearRegression  # type: ignore
except Exception:  # pragma: no cover - fallback when sklearn missing
    LinearRegression = None  # type: ignore

from .adaptive_roi_dataset import load_adaptive_roi_dataset

__all__ = ["AdaptiveROIPredictor", "predict_growth_type"]


class AdaptiveROIPredictor:
    """Train a lightweight model to forecast ROI and growth patterns."""

    def __init__(self) -> None:
        X, y, _ = load_adaptive_roi_dataset()
        if X.size and y.size:
            if GradientBoostingRegressor is not None:
                self._model = GradientBoostingRegressor(random_state=0)
            elif LinearRegression is not None:
                self._model = LinearRegression()
            else:  # pragma: no cover - extremely minimal environment
                self._model = None
            if self._model is not None:
                try:
                    self._model.fit(X, y)
                except Exception:  # pragma: no cover - dataset issues
                    self._model = None
        else:
            self._model = None

    # ------------------------------------------------------------------
    def _predict_sequence(self, features: np.ndarray) -> np.ndarray:
        """Return model predictions for ``features``.

        ``features`` must be a ``(n_samples, n_features)`` array representing
        sequential action observations.  When no model is available the first
        column is returned unchanged as a naive baseline.
        """

        if features.ndim != 2:
            raise ValueError("features must be 2D")
        if getattr(self._model, "predict", None) is not None:
            try:
                return np.asarray(self._model.predict(features), dtype=float)
            except Exception:  # pragma: no cover - prediction failure
                pass
        return features[:, 0].astype(float)

    # ------------------------------------------------------------------
    def predict_growth_type(self, action_features: Sequence[Sequence[float]]) -> str:
        """Classify the ROI growth trajectory for ``action_features``.

        Parameters
        ----------
        action_features:
            Sequence of feature vectors ordered chronologically.  Each vector
            should match the feature layout used by
            :func:`load_adaptive_roi_dataset` (currently ``[roi, delta]``).

        Returns
        -------
        str
            ``"exponential"``, ``"linear"`` or ``"marginal"`` depending on
            which curve best fits the predicted ROI sequence.
        """

        feats = np.asarray(list(action_features), dtype=float)
        if feats.size == 0:
            return "marginal"
        preds = self._predict_sequence(feats)
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


# Module‑level convenience instance -----------------------------------------
_predictor: AdaptiveROIPredictor | None = None


def predict_growth_type(action_features: Sequence[Sequence[float]]) -> str:
    """Return growth classification for ``action_features``.

    A singleton :class:`AdaptiveROIPredictor` instance is created lazily on
    first use to avoid unnecessary training when the API is imported but never
    called.
    """

    global _predictor
    if _predictor is None:
        _predictor = AdaptiveROIPredictor()
    return _predictor.predict_growth_type(action_features)
