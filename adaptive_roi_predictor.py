from __future__ import annotations

"""Predict ROI growth patterns and ROI estimates using a lightweight model.

This module trains a simple regression model on the aggregated ROI dataset
returned by :func:`adaptive_roi_dataset.build_dataset`.  Given a sequence of
improvement feature vectors it forecasts ROI over the provided horizon and
classifies the projected curve as ``"exponential"``, ``"linear"`` or
``"marginal"`` using slope and curvature thresholds.  Trained model
parameters are stored on disk to allow incremental retraining when new data
becomes available.
"""

from pathlib import Path
from typing import Sequence, Tuple, Dict, Any

import numpy as np
import json
import sqlite3

from dynamic_path_router import resolve_path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas missing
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        GradientBoostingClassifier,
    )  # type: ignore
except Exception:  # pragma: no cover - sklearn missing
    GradientBoostingRegressor = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LinearRegression, SGDRegressor  # type: ignore
    from sklearn.preprocessing import PolynomialFeatures  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
except Exception:  # pragma: no cover - sklearn missing
    LinearRegression = None  # type: ignore
    SGDRegressor = None  # type: ignore
    PolynomialFeatures = None  # type: ignore
    make_pipeline = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.multioutput import MultiOutputRegressor  # type: ignore
except Exception:  # pragma: no cover - sklearn missing
    MultiOutputRegressor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.model_selection import GridSearchCV, KFold  # type: ignore
except Exception:  # pragma: no cover - sklearn missing
    GridSearchCV = None  # type: ignore
    KFold = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.feature_selection import SelectKBest, f_regression  # type: ignore
except Exception:  # pragma: no cover - sklearn missing
    SelectKBest = None  # type: ignore
    f_regression = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover - joblib missing
    joblib = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except Exception:  # pragma: no cover - statsmodels missing
    ARIMA = None  # type: ignore

import pickle

if __package__:
    from .logging_utils import get_logger
    from .adaptive_roi_dataset import build_dataset, _label_growth
    from .roi_tracker import ROITracker
    from .evaluation_history_db import EvaluationHistoryDB
    from .evolution_history_db import EvolutionHistoryDB
    from .truth_adapter import TruthAdapter
else:  # pragma: no cover - fallback when executed outside package
    import sys
    from importlib import import_module
    from pathlib import Path

    _pkg_root = Path(__file__).resolve().parent
    _pkg_name = _pkg_root.name
    _parent = _pkg_root.parent
    if str(_parent) not in sys.path:
        sys.path.append(str(_parent))

    get_logger = import_module(f"{_pkg_name}.logging_utils").get_logger  # type: ignore[attr-defined]
    dataset_mod = import_module(f"{_pkg_name}.adaptive_roi_dataset")
    build_dataset = dataset_mod.build_dataset  # type: ignore[attr-defined]
    _label_growth = dataset_mod._label_growth  # type: ignore[attr-defined]
    ROITracker = import_module(f"{_pkg_name}.roi_tracker").ROITracker  # type: ignore[attr-defined]
    EvaluationHistoryDB = import_module(f"{_pkg_name}.evaluation_history_db").EvaluationHistoryDB  # type: ignore[attr-defined]
    EvolutionHistoryDB = import_module(f"{_pkg_name}.evolution_history_db").EvolutionHistoryDB  # type: ignore[attr-defined]
    TruthAdapter = import_module(f"{_pkg_name}.truth_adapter").TruthAdapter  # type: ignore[attr-defined]
from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

MENACE_ID = "adaptive_roi_predictor"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)

__all__ = [
    "AdaptiveROIPredictor",
    "predict_growth_type",
    "predict",
    "load_training_data",
]


logger = get_logger(__name__)


class ARIMARegressor:
    """Minimal wrapper around :class:`statsmodels` ARIMA for scikit API."""

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)) -> None:
        self.order = order
        self._model = None
        self._dim = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ARIMARegressor":  # pragma: no cover - thin wrapper
        if ARIMA is None:
            raise RuntimeError("statsmodels is required for ARIMARegressor")
        arr = np.asarray(y, dtype=float)
        if arr.ndim > 1:
            self._dim = arr.shape[1]
            arr = arr[:, 0]
        else:
            self._dim = 1
        self._model = ARIMA(arr, order=self.order).fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - thin wrapper
        if self._model is None:
            raise RuntimeError("model has not been fitted")
        steps = len(X)
        preds = np.asarray(self._model.forecast(steps=steps))
        if self._dim > 1:
            preds = np.tile(preds.reshape(-1, 1), (1, self._dim))
        else:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self) -> Dict[str, Any]:  # pragma: no cover - thin wrapper
        return {"order": self.order}

class AdaptiveROIPredictor:
    """Train a lightweight model to forecast ROI and growth patterns.

    The model is trained on the dataset produced by :func:`build_dataset` and
    persisted to ``model_path`` (default ``sandbox_data/adaptive_roi.pkl``) so
    that future instances can reuse the learned parameters without retraining
    from scratch.
    """

    def __init__(
        self,
        model_path: Path | str = resolve_path("sandbox_data/adaptive_roi.pkl"),
        cv: int = 3,
        param_grid: Dict[str, Dict[str, Any]] | None = None,
        slope_threshold: float | None = None,
        curvature_threshold: float | None = None,
    ) -> None:
        model_path = Path(model_path)
        try:
            self.model_path = resolve_path(model_path.as_posix())
        except FileNotFoundError:
            self.model_path = resolve_path(model_path.parent.as_posix()) / model_path.name
        self.cv = cv
        param_grid_provided = param_grid is not None
        default_grid = {
            "GradientBoostingRegressor": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
            },
            "GradientBoostingClassifier": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
            },
            "SGDRegressor": {
                "sgdregressor__alpha": [0.0001, 0.001],
                "sgdregressor__penalty": ["l2", "l1"],
            },
            "ARIMARegressor": {
                "order": [(1, 0, 0), (2, 1, 0)]
            },
        }
        self.param_grid: Dict[str, Dict[str, Any]] = param_grid or default_grid
        self._param_grid_provided = param_grid_provided
        self._model = None
        self._classifier = None
        self.best_params: Dict[str, Any] | None = None
        self.best_score: float | None = None
        self.validation_scores: Dict[str, float] = {}
        self.slope_threshold: float | None = slope_threshold
        self.curvature_threshold: float | None = curvature_threshold
        self.training_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._trained_size: int = 0
        self.selected_features: list[str] | None = None
        self.drift_metrics: Dict[str, float] = {}
        self.truth_adapter = TruthAdapter()
        self._load()
        if slope_threshold is not None:
            self.slope_threshold = slope_threshold
        if curvature_threshold is not None:
            self.curvature_threshold = curvature_threshold
        if self._model is None or self._param_grid_provided:
            self.train(cv=self.cv, param_grid=self.param_grid)

    # ------------------------------------------------------------------
    # persistence helpers
    def _load(self) -> None:
        """Load model parameters from ``self.model_path`` if available."""

        if self.model_path.exists():
            try:
                if joblib is not None:
                    obj = joblib.load(self.model_path)
                else:
                    with self.model_path.open("rb") as fh:
                        obj = pickle.load(fh)
                if isinstance(obj, dict) and "model" in obj:
                    self._model = obj.get("model")
                    self._classifier = obj.get("classifier")
                    data = obj.get("data")
                    if isinstance(data, tuple) and len(data) in (2, 3):
                        self.training_data = tuple(np.asarray(d) for d in data)  # type: ignore
                    self._trained_size = int(obj.get("n_samples", 0))
                else:
                    self._model = obj
            except Exception:  # pragma: no cover - corrupted file
                self._model = None
                self._classifier = None
                self.training_data = None
                self._trained_size = 0

        meta_path = self.model_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                self.best_params = data.get("best_params")
                self.best_score = data.get("best_score")
                self.validation_scores = data.get("validation_scores", {}) or {}
                self.slope_threshold = data.get(
                    "slope_threshold", self.slope_threshold
                )
                self.curvature_threshold = data.get(
                    "curvature_threshold", self.curvature_threshold
                )
                sel = data.get("selected_features")
                if isinstance(sel, list):
                    self.selected_features = [str(s) for s in sel]
                self.drift_metrics = data.get("drift_metrics", {}) or {}
            except Exception:
                self.best_params = None
                self.best_score = None
                self.validation_scores = {}

    def _save(self) -> None:
        """Persist the current model to disk."""

        if self._model is None and self._classifier is None:
            # Even when no model was produced we persist metadata so that
            # cross‑validation results from failed runs can be inspected or
            # reused on the next attempt.
            self._save_meta()
            return
        try:  # pragma: no cover - disk issues
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": self._model,
                "classifier": self._classifier,
                "data": self.training_data,
                "n_samples": self._trained_size,
            }
            if joblib is not None:
                joblib.dump(payload, self.model_path)
            else:
                with self.model_path.open("wb") as fh:
                    pickle.dump(payload, fh)
        except Exception:
            pass
        self._save_meta()

    def _save_meta(self) -> None:
        """Persist metadata about the trained model."""

        try:  # pragma: no cover - disk issues
            meta = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "validation_scores": self.validation_scores,
                "slope_threshold": self.slope_threshold,
                "curvature_threshold": self.curvature_threshold,
                "selected_features": self.selected_features,
                "drift_metrics": self.drift_metrics,
            }
            meta_path = self.model_path.with_suffix(".meta.json")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def record_drift(
        self,
        accuracy: float,
        mae: float,
        *,
        acc_threshold: float = 0.6,
        mae_threshold: float = 0.1,
        retrain: bool = False,
    ) -> None:
        """Store the most recent drift metrics and handle retraining.

        Parameters
        ----------
        accuracy, mae:
            Latest evaluation metrics.
        acc_threshold, mae_threshold:
            Bounds outside of which model drift is assumed.
        retrain:
            When ``True`` and thresholds are violated the model is updated via
            :meth:`partial_fit` or a full :meth:`train`.
        """

        self.drift_metrics = {"accuracy": float(accuracy), "mae": float(mae)}
        try:  # pragma: no cover - disk issues
            self._save_meta()
        except Exception:
            pass

        if not retrain:
            return

        if accuracy < acc_threshold or mae > mae_threshold:
            try:
                self.partial_fit()
                logger.info(
                    "adaptive ROI model retrained (acc=%.3f, mae=%.3f)",
                    accuracy,
                    mae,
                )
            except Exception:
                logger.exception("incremental update failed, full retrain")
                try:
                    self.train()
                    logger.info(
                        "adaptive ROI model retrained from scratch (acc=%.3f, mae=%.3f)",
                        accuracy,
                        mae,
                    )
                except Exception:
                    logger.exception("full retrain failed")

    # ------------------------------------------------------------------
    def partial_fit(
        self,
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        *,
        batch_size: int = 32,
    ) -> None:
        """Incrementally update the model with a mini-batch of new data."""

        if dataset is None:
            X, y, g = build_dataset()
        else:
            X, y, g = dataset

        if self._model is None or not hasattr(self._model, "partial_fit"):
            # Fallback to full retraining when incremental updates are unsupported
            self.train((X, y, g))
            return

        start = max(self._trained_size, len(X) - batch_size)
        try:
            self._model.partial_fit(X[start:], y[start:])
            if self._classifier is not None and hasattr(self._classifier, "partial_fit"):
                classes = np.unique(g[start:]) if self._trained_size == 0 else None
                kwargs: Dict[str, Any] = {"classes": classes} if classes is not None else {}
                self._classifier.partial_fit(X[start:], g[start:], **kwargs)
            self._trained_size = len(X)
            window = 200
            self.training_data = (X[-window:], y[-window:], g[-window:])
            self._save()
        except Exception:
            # If incremental update fails perform a full retrain
            self.train((X, y, g))

    # ------------------------------------------------------------------
    def update(
        self,
        X_new: np.ndarray | Sequence[Sequence[float]],
        y_new: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
        g_new: np.ndarray | Sequence[str],
    ) -> None:
        """Append ``*_new`` samples and update the model online.

        Parameters
        ----------
        X_new, y_new, g_new:
            New feature rows, ROI targets and growth labels.  The arrays must
            have matching lengths.  When the underlying model implements
            :meth:`partial_fit` the new samples are used to incrementally
            refine the existing model.  Otherwise a full retraining on all
            accumulated samples is triggered.
        """

        X_arr = np.asarray(X_new, dtype=float)
        y_arr = np.asarray(y_new, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(len(y_arr), -1)
        g_arr = np.asarray(g_new, dtype=object)

        if self.training_data is not None:
            X, y, g = self.training_data
            X = np.vstack([X, X_arr])
            y = np.vstack([y, y_arr])
            g = np.concatenate([g, g_arr])
        else:
            X, y, g = X_arr, y_arr, g_arr

        # ``partial_fit`` handles detection of incremental capability and
        # falls back to full retraining when unsupported.
        self.partial_fit((X, y, g), batch_size=len(X_arr))

    # ------------------------------------------------------------------
    def calibrate_thresholds(
        self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> Tuple[float, float]:
        """Estimate slope and curvature cutoffs from ``dataset``.

        The thresholds are derived from the distribution of first and second
        derivatives of the ROI targets.  When ``dataset`` is ``None`` the most
        recently stored ``self.training_data`` is used.  The resulting
        thresholds are stored on the instance and returned.
        """

        if dataset is None:
            dataset = self.training_data
        if dataset is None:
            self.slope_threshold = self.slope_threshold or 0.05
            self.curvature_threshold = self.curvature_threshold or 0.01
            return float(self.slope_threshold), float(self.curvature_threshold)

        _X, y, _g = dataset
        arr = np.asarray(y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        first = np.diff(arr, axis=1)
        slope_vals = np.abs(first).ravel()
        slope_thr = float(np.percentile(slope_vals, 75)) if slope_vals.size else 0.05
        second = np.diff(first, axis=1)
        curv_vals = np.abs(second).ravel()
        curv_thr = float(np.percentile(curv_vals, 75)) if curv_vals.size else 0.01
        self.slope_threshold = slope_thr
        self.curvature_threshold = curv_thr
        return slope_thr, curv_thr

    # ------------------------------------------------------------------
    # training
    def train(
        self,
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        cv: int | None = None,
        param_grid: Dict[str, Dict[str, Any]] | None = None,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        """Fit the underlying model on ``dataset``.

        Parameters
        ----------
        dataset:
            Optional tuple ``(features, targets, growth_types)`` where ``targets``
            may contain multiple horizon columns.  When omitted the data is
            loaded using :func:`build_dataset`.
        cv:
            Number of cross-validation folds.  When ``None`` the instance's
            default is used.  Values less than 2 disable cross-validation.
        param_grid:
            Optional mapping of model names to parameter grids for
            :class:`~sklearn.model_selection.GridSearchCV`.
        """

        if dataset is None:
            X, y, g, names = build_dataset(return_feature_names=True)
            feature_names = names
        else:
            X, y, g = dataset

        if self.slope_threshold is None or self.curvature_threshold is None:
            self.calibrate_thresholds((X, y, g))

        if X.size == 0 or y.size == 0:
            self._model = None
            self._classifier = None
            self.training_data = None
            self._trained_size = 0
            return

        total_size = len(X)
        window = 200
        self.training_data = (X[-window:], y[-window:], g[-window:])
        multi_output = y.ndim == 2 and y.shape[1] > 1

        if (
            self._model is not None
            and hasattr(self._model, "partial_fit")
            and self._trained_size > 0
            and total_size > self._trained_size
        ):
            try:
                self._model.partial_fit(
                    X[self._trained_size :], y[self._trained_size :]
                )
                if self._classifier is not None and hasattr(self._classifier, "partial_fit"):
                    self._classifier.partial_fit(
                        X[self._trained_size :], g[self._trained_size :]
                    )
                elif GradientBoostingClassifier is not None:
                    self._classifier = GradientBoostingClassifier(random_state=0)
                    self._classifier.fit(X, g)
                self._trained_size = total_size
                self.training_data = (X[-window:], y[-window:], g[-window:])
                self._save()
                return
            except Exception:  # pragma: no cover - incremental fit failure
                pass

        cv = (cv if cv is not None else self.cv) or 0
        param_grid = param_grid if param_grid is not None else self.param_grid

        candidates: list[tuple[str, Any]] = []
        if GradientBoostingRegressor is not None:
            params: Dict[str, Any] = {}
            if self.best_params and self.best_params.get("model") == "GradientBoostingRegressor":
                params = {k: v for k, v in self.best_params.items() if k != "model"}
            model: Any = GradientBoostingRegressor(random_state=0, **params)
            if multi_output and MultiOutputRegressor is not None:
                model = MultiOutputRegressor(model)
            candidates.append(("GradientBoostingRegressor", model))
        if (
            SGDRegressor is not None
            and PolynomialFeatures is not None
            and make_pipeline is not None
        ):
            params = {}
            if self.best_params and self.best_params.get("model") == "SGDRegressor":
                params = {k: v for k, v in self.best_params.items() if k != "model"}
            base = make_pipeline(
                PolynomialFeatures(degree=2),
                SGDRegressor(max_iter=1_000, tol=1e-3, random_state=0, **params),
            )
            model = base
            if multi_output and MultiOutputRegressor is not None:
                model = MultiOutputRegressor(base)
            candidates.append(("SGDRegressor", model))
        if LinearRegression is not None:
            params = {}
            if self.best_params and self.best_params.get("model") == "LinearRegression":
                params = {k: v for k, v in self.best_params.items() if k != "model"}
            model = LinearRegression(**params)
            if multi_output and MultiOutputRegressor is not None:
                model = MultiOutputRegressor(model)
            candidates.append(("LinearRegression", model))
        if ARIMA is not None:
            params = {}
            if self.best_params and self.best_params.get("model") == "ARIMARegressor":
                params = {k: v for k, v in self.best_params.items() if k != "model"}
            model = ARIMARegressor(**params)
            candidates.append(("ARIMARegressor", model))

        best_model = None
        best_score = float("inf")
        best_params: Dict[str, Any] | None = None
        self.validation_scores = {}

        for name, model in candidates:
            grid = param_grid.get(name, {}) if isinstance(param_grid, dict) else {}
            score = float("inf")
            params: Dict[str, Any] = {"model": name}

            if name == "ARIMARegressor":
                order_grid = grid.get("order", [(1, 0, 0)]) if isinstance(grid, dict) else [(1, 0, 0)]
                for order in order_grid:
                    try:
                        m = ARIMARegressor(order=order)
                        m.fit(X, y)
                        preds = m.predict(X)
                        sc = float(np.mean(np.abs(preds - y)))
                        if sc < score:
                            score = sc
                            model = m
                            params.update({"order": order})
                    except Exception:  # pragma: no cover - arima failure
                        continue
            elif GridSearchCV is not None and grid and len(X) >= max(cv, 2):
                try:
                    gs = GridSearchCV(
                        model,
                        grid,
                        cv=cv,
                        scoring="neg_mean_absolute_error",
                    )
                    gs.fit(X, y)
                    score = float(-gs.best_score_)
                    model = gs.best_estimator_
                    params.update(gs.best_params_)
                except Exception:  # pragma: no cover - grid search failure
                    pass
            elif KFold is not None and cv > 1 and len(X) >= cv:
                kf = KFold(n_splits=cv, shuffle=True, random_state=0)
                errors: list[float] = []
                for train_idx, test_idx in kf.split(X):
                    m = pickle.loads(pickle.dumps(model))
                    try:
                        m.fit(X[train_idx], y[train_idx])
                        preds = m.predict(X[test_idx])
                        errors.append(float(np.mean(np.abs(preds - y[test_idx]))))
                    except Exception:  # pragma: no cover - fit failure
                        errors.append(float("inf"))
                score = float(np.mean(errors))
                try:
                    model.fit(X, y)
                except Exception:  # pragma: no cover - fit failure
                    continue
                params.update(getattr(model, "get_params", lambda: {})())
            else:
                try:
                    model.fit(X, y)
                    preds = model.predict(X)
                    score = float(np.mean(np.abs(preds - y)))
                    params.update(getattr(model, "get_params", lambda: {})())
                except Exception:  # pragma: no cover - fit failure
                    continue

            self.validation_scores[name] = score
            if score < best_score:
                best_score = score
                best_model = model
                best_params = params

        self._model = best_model
        self.best_score = None if best_score == float("inf") else best_score
        self.best_params = best_params
        self._classifier = None
        clf_best_params: Dict[str, Any] | None = None
        if g.size and GradientBoostingClassifier is not None:
            try:
                clf_params: Dict[str, Any] = {}
                if (
                    self.best_params
                    and isinstance(self.best_params.get("classifier"), dict)
                ):
                    clf_params = {
                        k: v
                        for k, v in self.best_params["classifier"].items()
                        if k != "model"
                    }
                clf = GradientBoostingClassifier(random_state=0, **clf_params)
                clf_grid = (
                    param_grid.get("GradientBoostingClassifier", {})
                    if isinstance(param_grid, dict)
                    else {}
                )
                if (
                    GridSearchCV is not None
                    and clf_grid
                    and len(g) >= max(cv, 2)
                    and len(np.unique(g)) > 1
                ):
                    gs_clf = GridSearchCV(
                        clf,
                        clf_grid,
                        cv=cv,
                        scoring="accuracy",
                    )
                    gs_clf.fit(X, g)
                    clf = gs_clf.best_estimator_
                    clf_best_params = {"model": "GradientBoostingClassifier"}
                    clf_best_params.update(gs_clf.best_params_)
                else:
                    clf.fit(X, g)
                    clf_best_params = {"model": "GradientBoostingClassifier"}
                    clf_best_params.update(
                        getattr(clf, "get_params", lambda: {})()
                    )
                self._classifier = clf
            except Exception:  # pragma: no cover - classifier failure
                self._classifier = None
                clf_best_params = None
        if clf_best_params is not None:
            if self.best_params is None:
                self.best_params = {}
            self.best_params["classifier"] = clf_best_params
        if self._model is not None and feature_names is not None:
            selected: list[str] | None = None
            if SelectKBest is not None and f_regression is not None:
                try:
                    y_sel = y[:, 0] if y.ndim > 1 else y
                    k = min(len(feature_names), 20)
                    selector = SelectKBest(score_func=f_regression, k=k)
                    selector.fit(X, y_sel)
                    idx = selector.get_support(indices=True)
                    selected = [feature_names[i] for i in idx]
                except Exception:
                    selected = None
            if selected is None and hasattr(self._model, "feature_importances_"):
                try:
                    importances = np.asarray(getattr(self._model, "feature_importances_"))
                    order = np.argsort(importances)[::-1][: min(len(importances), 20)]
                    selected = [feature_names[i] for i in order]
                except Exception:
                    selected = None
            self.selected_features = selected
        if self._model is not None:
            try:
                self._trained_size = total_size
                self._save()
            except Exception:  # pragma: no cover - training failure
                self._model = None
                self._trained_size = 0

    # ------------------------------------------------------------------
    def _predict_sequence(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return model predictions and confidence for ``features``.

        ``features`` must be a ``(n_samples, n_features)`` array.  When an
        ensemble of models is present, the mean prediction across the ensemble
        is returned and the variance is used to derive a confidence score for
        each forecast.  Confidence is mapped to ``[0, 1]`` using ``1/(1+var)``
        so that agreement between models yields a value near ``1`` while high
        disagreement trends towards ``0``.  When the model is unavailable the
        first feature column is used as a naive baseline with zero confidence.
        """

        if features.ndim != 2:
            raise ValueError("features must be 2D")

        # Support ensembles of models (e.g. bootstrap aggregations)
        if isinstance(self._model, (list, tuple)) and self._model:
            preds: list[np.ndarray] = []
            for model in self._model:
                if getattr(model, "predict", None) is None:
                    continue
                try:  # pragma: no cover - best effort prediction
                    preds.append(np.asarray(model.predict(features), dtype=float))
                except Exception:
                    continue
            if preds:
                stacked = np.stack(preds, axis=0)
                mean = stacked.mean(axis=0)
                var = stacked.var(axis=0)
                confidence = 1.0 / (1.0 + var)
                return mean, confidence

        if getattr(self._model, "predict", None) is not None:
            try:
                pred = np.asarray(self._model.predict(features), dtype=float)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                conf = np.ones_like(pred, dtype=float)
                return pred, conf
            except Exception:  # pragma: no cover - prediction failure
                pass

        baseline = features[:, 0].astype(float)
        if baseline.ndim == 1:
            baseline = baseline.reshape(-1, 1)
        n_targets = 1
        if self.training_data is not None and self.training_data[1].ndim == 2:
            n_targets = self.training_data[1].shape[1]
        if baseline.shape[1] < n_targets:
            baseline = np.repeat(baseline, n_targets, axis=1)
        return baseline, np.zeros_like(baseline)

    # ------------------------------------------------------------------
    # public API
    def predict(
        self,
        improvement_features: Sequence[Sequence[float]],
        horizon: int | None = None,
        *,
        tracker: ROITracker | None = None,
        actual_roi: Sequence[float] | float | None = None,
        actual_class: str | None = None,
    ) -> tuple[list[list[float]], str, list[list[float]], float | None]:
        """Return ROI forecast sequences, growth classification and confidence.

        Parameters
        ----------
        improvement_features:
            Sequence of feature vectors describing consecutive improvement
            cycles.
        horizon:
            Number of steps to forecast. Defaults to the length of
            ``improvement_features``. Only the first ``horizon`` rows of
            ``improvement_features`` are used.
        tracker:
            Optional :class:`ROITracker` used to record prediction events.
        actual_roi, actual_class:
            When provided, the prediction along with the actual outcome and
            class label are logged through ``tracker`` and drift metrics are
            updated.

        Returns
        -------
        list, str, list, float
        The predicted ROI values for each step (possibly multiple horizons per
        step), the growth classification from the final step, per-step
        confidence values in the range ``[0, 1]`` with the same shape as the
        predictions, and the classifier probability for the growth label when
        available.
        """

        feats = np.asarray(list(improvement_features), dtype=float)
        h = len(feats) if horizon is None else int(horizon)
        if h <= 0:
            return [], "marginal", [], None
        feats = feats[:h]
        preds, conf = self._predict_sequence(feats)
        try:
            realish, low_conf = self.truth_adapter.predict(feats)
            preds = np.asarray(realish, dtype=float).reshape(preds.shape)
            if low_conf:
                logger.warning("truth adapter low confidence; consider retraining")
        except Exception:
            logger.exception("truth adapter predict failed")
        growth: str
        growth_conf: float | None = None
        if self._classifier is not None and getattr(self._classifier, "predict", None) is not None:
            try:
                growth = str(self._classifier.predict(feats)[-1])
                if getattr(self._classifier, "predict_proba", None) is not None:
                    try:
                        probs = self._classifier.predict_proba(feats)[-1]
                        labels = getattr(self._classifier, "classes_", None)
                        if labels is not None and growth in labels:
                            idx = list(labels).index(growth)
                            growth_conf = float(probs[idx])
                        else:
                            growth_conf = float(max(probs))
                    except Exception:  # pragma: no cover - probability failure
                        growth_conf = None
            except Exception:  # pragma: no cover - classifier prediction failure
                growth = "marginal"
        else:
            growth = "marginal"
        result = (preds.tolist(), growth, conf.tolist(), growth_conf)
        if tracker is not None and actual_roi is not None:
            try:
                tracker.record_prediction(
                    preds[-1].tolist() if preds.size else [],
                    actual_roi,
                    predicted_class=growth,
                    actual_class=actual_class,
                    confidence=growth_conf,
                )
                mae = tracker.rolling_mae()
                acc = tracker.classification_accuracy()
                self.record_drift(acc, mae)
            except Exception:
                pass
        return result

    # Backwards compatible wrapper
    def predict_growth_type(
        self, action_features: Sequence[Sequence[float]], horizon: int | None = None
    ) -> str:
        """Return only the growth classification for ``action_features``."""

        return self.predict(action_features, horizon=horizon)[1]

    # ------------------------------------------------------------------
    def evaluate_model(
        self,
        tracker: ROITracker,
        *,
        accuracy_threshold: float = 0.6,
        mae_threshold: float = 0.1,
        retrain: bool = True,
        **kwargs,
    ) -> tuple[float, float]:
        """Evaluate prediction accuracy and handle drift.

        Parameters
        ----------
        tracker:
            ROITracker instance providing evaluation history.
        accuracy_threshold, mae_threshold:
            Bounds outside of which drift is assumed and the model retrained.
        retrain:
            When ``True`` retraining is triggered automatically if thresholds
            are exceeded.
        kwargs:
            Additional parameters forwarded to :meth:`ROITracker.evaluate_model`.
        """

        acc, mae = tracker.evaluate_model(
            mae_threshold=mae_threshold,
            acc_threshold=accuracy_threshold,
            **kwargs,
        )
        self.record_drift(
            acc,
            mae,
            acc_threshold=accuracy_threshold,
            mae_threshold=mae_threshold,
            retrain=retrain,
        )
        return acc, mae


# Module‑level convenience instance -----------------------------------------
_predictor: AdaptiveROIPredictor | None = None


def predict_growth_type(
    action_features: Sequence[Sequence[float]], horizon: int | None = None
) -> str:
    """Return growth classification for ``action_features`` using a singleton."""

    global _predictor
    if _predictor is None:
        _predictor = AdaptiveROIPredictor()
    return _predictor.predict_growth_type(action_features, horizon=horizon)


def predict(
    action_features: Sequence[Sequence[float]],
    horizon: int | None = None,
    *,
    tracker: ROITracker | None = None,
    actual_roi: Sequence[float] | float | None = None,
    actual_class: str | None = None,
) -> tuple[list[list[float]], str, list[list[float]], float | None]:
    """Return ROI forecast sequences, growth category and confidence using a module-level predictor."""

    global _predictor
    if _predictor is None:
        _predictor = AdaptiveROIPredictor()
    return _predictor.predict(
        action_features,
        horizon=horizon,
        tracker=tracker,
        actual_roi=actual_roi,
        actual_class=actual_class,
    )


def load_training_data(
    tracker: ROITracker,
    evolution_path: str | Path = "evolution_history.db",
    roi_events_path: str | Path = "roi_events.db",
    output_path: Path | str = resolve_path("sandbox_data/adaptive_roi.csv"),
    *,
    router: DBRouter | None = None,
) -> "pd.DataFrame":
    """Collect and normalise ROI training data.

    Parameters
    ----------
    tracker:
        :class:`ROITracker` instance providing in-memory histories.
    evolution_path:
        Path to the evolution history database supplying ROI outcome labels.
    roi_events_path:
        Path to the ROI event log database used for additional ROI deltas.
    output_path:
        CSV file where the assembled dataset will be written.
    router:
        Optional :class:`DBRouter` instance for database access.

    Returns
    -------
    pandas.DataFrame
        The merged and normalised dataset.  Requires :mod:`pandas` to be
        available.
    """

    if pd is None:  # pragma: no cover - pandas not installed
        raise RuntimeError("pandas is required for load_training_data")

    evolution_path = Path(evolution_path)
    roi_events_path = Path(roi_events_path)
    output_path = Path(output_path)

    try:
        evolution_path = resolve_path(evolution_path.as_posix())
    except FileNotFoundError:
        evolution_path = resolve_path(evolution_path.parent.as_posix()) / evolution_path.name
    try:
        roi_events_path = resolve_path(roi_events_path.as_posix())
    except FileNotFoundError:
        roi_events_path = resolve_path(roi_events_path.parent.as_posix()) / roi_events_path.name
    try:
        output_path = resolve_path(output_path.as_posix())
    except FileNotFoundError:
        output_path = resolve_path(output_path.parent.as_posix()) / output_path.name

    router = router or DB_ROUTER

    n = len(tracker.roi_history)
    data: dict[str, list[float]] = {
        "roi_delta": [float(x) for x in tracker.roi_history]
    }
    for name, vals in tracker.metrics_history.items():
        seq = [float(v) for v in vals]
        if len(seq) < n:
            seq.extend([0.0] * (n - len(seq)))
        data[name] = seq[:n]
    # include synergy metrics ------------------------------------------------
    for name, vals in tracker.synergy_metrics_history.items():
        seq = [float(v) for v in vals]
        if len(seq) < n:
            seq.extend([0.0] * (n - len(seq)))
        data[name] = seq[:n]
    # resource usage metrics -------------------------------------------------
    res_cols = ["cpu", "memory", "disk", "time", "gpu"]
    res_data = {c: [] for c in res_cols}
    for row in tracker.resource_metrics[:n]:
        for c, val in zip(res_cols, row):
            res_data[c].append(float(val))
    for c in res_cols:
        if len(res_data[c]) < n:
            res_data[c].extend([0.0] * (n - len(res_data[c])))
        data[c] = res_data[c][:n]
    df = pd.DataFrame(data)

    # GPT evaluation scores -------------------------------------------------
    eval_db = EvaluationHistoryDB(router=router)
    recs: list[tuple[pd.Timestamp, float]] = []
    for eng in eval_db.engines():
        for score, ts, _passed, _err in eval_db.history(eng, limit=1_000_000):
            recs.append((pd.to_datetime(ts), float(score)))
    recs.sort(key=lambda r: r[0])
    scores = [r[1] for r in recs[:n]]
    if len(scores) < n:
        scores.extend([0.0] * (n - len(scores)))
    df["gpt_score"] = scores

    # ROI event deltas ------------------------------------------------------
    try:
        with sqlite3.connect(roi_events_path) as conn:
            try:
                roi_df = pd.read_sql(
                    "SELECT roi_after - roi_before AS roi_event_delta FROM roi_events ORDER BY ts",
                    conn,
                )
            except Exception:  # pragma: no cover - missing table or DB
                roi_df = pd.DataFrame(columns=["roi_event_delta"])
            try:
                pred_df = pd.read_sql(
                    "SELECT predicted_roi, actual_roi, predicted_class, actual_class, confidence FROM roi_prediction_events ORDER BY ts",
                    conn,
                )
            except Exception:  # pragma: no cover - missing table or DB
                pred_df = pd.DataFrame(
                    columns=[
                        "predicted_roi",
                        "actual_roi",
                        "predicted_class",
                        "actual_class",
                        "confidence",
                    ]
                )
    except Exception:
        roi_df = pd.DataFrame(columns=["roi_event_delta"])
        pred_df = pd.DataFrame(
            columns=[
                "predicted_roi",
                "actual_roi",
                "predicted_class",
                "actual_class",
                "confidence",
            ]
        )

    event_deltas = roi_df.get("roi_event_delta", pd.Series(dtype=float)).astype(float).tolist()
    if len(event_deltas) < n:
        event_deltas.extend([0.0] * (n - len(event_deltas)))
    df["roi_event_delta"] = event_deltas[:n]

    # Persisted prediction history -----------------------------------------
    pred_vals = pred_df.get("predicted_roi", pd.Series(dtype=float)).astype(float).tolist()
    act_vals = pred_df.get("actual_roi", pd.Series(dtype=float)).astype(float).tolist()
    pred_classes = pd.Categorical(
        pred_df.get("predicted_class", pd.Series(dtype=str))
    ).codes.tolist()
    act_classes = pd.Categorical(
        pred_df.get("actual_class", pd.Series(dtype=str))
    ).codes.tolist()
    conf_vals = pred_df.get("confidence", pd.Series(dtype=float)).astype(float).tolist()
    if len(pred_vals) < n:
        pred_vals.extend([0.0] * (n - len(pred_vals)))
    if len(act_vals) < n:
        act_vals.extend([0.0] * (n - len(act_vals)))
    if len(pred_classes) < n:
        pred_classes.extend([0] * (n - len(pred_classes)))
    if len(act_classes) < n:
        act_classes.extend([0] * (n - len(act_classes)))
    if len(conf_vals) < n:
        conf_vals.extend([0.0] * (n - len(conf_vals)))
    df["predicted_roi_event"] = pred_vals[:n]
    df["actual_roi_event"] = act_vals[:n]
    df["predicted_class_event"] = pred_classes[:n]
    df["actual_class_event"] = act_classes[:n]
    df["prediction_confidence"] = conf_vals[:n]

    # ROI outcome labels ----------------------------------------------------
    evo_db = EvolutionHistoryDB(evolution_path)
    events = sorted(evo_db.fetch(limit=1_000_000), key=lambda r: r[9])
    outcomes = [float(ev[3]) for ev in events[:n]]
    if len(outcomes) < n:
        outcomes.extend([0.0] * (n - len(outcomes)))
    df["roi_outcome"] = outcomes

    # Growth labels ---------------------------------------------------------
    growth_seq: list[str] = []
    roi_vals: list[float] = []
    for val in tracker.roi_history[:n]:
        roi_vals.append(float(val))
        growth_seq.append(_label_growth(roi_vals))
    if len(growth_seq) < n:
        growth_seq.extend(["marginal"] * (n - len(growth_seq)))
    df["growth_label"] = pd.Categorical(growth_seq[:n]).codes

    # Normalise feature columns --------------------------------------------
    for col in df.columns:
        if col == "roi_outcome":
            continue
        series = df[col]
        if series.empty:
            continue
        mean = float(series.mean())
        std = float(series.std())
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        df[col] = (series - mean) / std

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

