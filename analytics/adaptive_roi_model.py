from __future__ import annotations

"""Train and apply a GradientBoostingRegressor for ROI forecasting.

This module trains a model on data provided by
:mod:`analytics.adaptive_roi_dataset` and persists the fitted model under
``analytics/models``.  The resulting forecaster returns both a numerical ROI
forecast and a qualitative growth label:
``"exponential"`` for high growth, ``"linear"`` for moderate growth and
``"marginal"`` for low or negative growth.
"""

import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - executed in minimal envs
    joblib = None  # type: ignore[assignment]
    _MISSING_JOBLIB = exc
else:
    _MISSING_JOBLIB = None

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - executed in minimal envs
    pd = None  # type: ignore[assignment]
    _MISSING_PANDAS = exc
else:
    _MISSING_PANDAS = None

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - executed in minimal envs
    GradientBoostingRegressor = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]
    r2_score = None  # type: ignore[assignment]
    _MISSING_SKLEARN = exc
else:
    _MISSING_SKLEARN = None

from .adaptive_roi_dataset import build_dataset
from dynamic_path_router import resolve_path

_LOGGER = logging.getLogger(__name__)
_DEPENDENCY_WARNING_EMITTED = False


def _dependency_error() -> ModuleNotFoundError:
    missing = []
    if _MISSING_JOBLIB is not None:
        missing.append("joblib")
    if _MISSING_PANDAS is not None:
        missing.append("pandas")
    if _MISSING_SKLEARN is not None:
        missing.append("scikit-learn")
    message = (
        "adaptive ROI modelling requires optional dependencies: "
        + ", ".join(missing)
        + ". Install menace_sandbox[analytics] or add the missing packages."
    )
    exc = ModuleNotFoundError(message)
    for original in (_MISSING_JOBLIB, _MISSING_PANDAS, _MISSING_SKLEARN):
        if original is not None:
            exc.__cause__ = original
            break
    return exc


def _ensure_dependencies() -> None:
    global _DEPENDENCY_WARNING_EMITTED
    if any(dep is not None for dep in (_MISSING_JOBLIB, _MISSING_PANDAS, _MISSING_SKLEARN)):
        if not _DEPENDENCY_WARNING_EMITTED:
            _LOGGER.warning(
                "adaptive ROI model disabled; optional dependencies missing"
            )
            _DEPENDENCY_WARNING_EMITTED = True
        raise _dependency_error()

# Path where the trained model is stored
MODEL_DIR = resolve_path("analytics/models")
MODEL_PATH = MODEL_DIR / "adaptive_roi_gbr.joblib"


def train(save_path: Path | str = MODEL_PATH) -> GradientBoostingRegressor:
    """Train the ROI forecasting model and persist it.

    Parameters
    ----------
    save_path:
        Location where the trained model should be written.  Defaults to
        :data:`MODEL_PATH`.
    """
    _ensure_dependencies()
    assert GradientBoostingRegressor is not None  # for type checkers
    assert joblib is not None
    assert pd is not None
    save_path = Path(save_path)
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


def retrain(save_path: Path | str = MODEL_PATH) -> Dict[str, float]:
    """Retrain the ROI forecasting model and return training metrics.

    Parameters
    ----------
    save_path:
        Location where the trained model should be written.  Defaults to
        :data:`MODEL_PATH`.
    """

    _ensure_dependencies()
    assert GradientBoostingRegressor is not None
    assert joblib is not None
    assert pd is not None
    save_path = Path(save_path)
    data = build_dataset()
    if data.empty:
        raise RuntimeError("Training dataset is empty")

    features = data[["performance_delta", "gpt_score"]]
    target = data["roi_delta"]

    model = GradientBoostingRegressor(random_state=0)
    model.fit(features, target)

    preds = model.predict(features)
    metrics = {
        "mse": float(mean_squared_error(target, preds)),
        "r2": float(r2_score(target, preds)),
        "n_samples": float(len(target)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return metrics


def load(model_path: Path | str = MODEL_PATH) -> GradientBoostingRegressor:
    """Load a previously trained model."""
    _ensure_dependencies()
    assert joblib is not None
    return joblib.load(Path(model_path))


def classify_growth(pred_series: Sequence[float]) -> str:
    """Classify growth trend for a sequence of ROI predictions.

    The classification is based on the average ratio of consecutive
    predictions.  Ratios above ``1.5`` indicate ``"exponential"`` growth
    while ratios above ``1.05`` but below ``1.5`` correspond to
    ``"linear"`` growth.  Any other pattern is considered ``"marginal"``.

    When ``pred_series`` contains a single value a fallback classification
    is performed using the raw ROI threshold: values above ``0.75`` are
    treated as ``"exponential"`` and positive values as ``"linear"``.
    """

    if not pred_series:
        return "marginal"

    if len(pred_series) == 1:
        roi = pred_series[0]
        if roi > 0.75:
            return "exponential"
        if roi > 0.0:
            return "linear"
        return "marginal"

    ratios = []
    prev = pred_series[0]
    for curr in pred_series[1:]:
        if prev != 0:
            ratios.append(curr / prev)
        prev = curr

    if not ratios:
        # All previous values were zero; fall back to last ROI value
        roi = pred_series[-1]
        if roi > 0.75:
            return "exponential"
        if roi > 0.0:
            return "linear"
        return "marginal"

    avg_ratio = sum(ratios) / len(ratios)
    if avg_ratio >= 1.5:
        return "exponential"
    if avg_ratio >= 1.05:
        return "linear"
    return "marginal"


def forecast(
    model: GradientBoostingRegressor,
    performance_delta: float,
    gpt_score: float,
) -> Tuple[float, str]:
    """Return ROI forecast and growth label for a single data point."""

    _ensure_dependencies()
    assert pd is not None
    df = pd.DataFrame(
        [[performance_delta, gpt_score]],
        columns=["performance_delta", "gpt_score"],
    )
    roi = float(model.predict(df)[0])
    return roi, classify_growth([roi])


__all__ = [
    "MODEL_DIR",
    "MODEL_PATH",
    "train",
    "retrain",
    "load",
    "forecast",
    "classify_growth",
]
