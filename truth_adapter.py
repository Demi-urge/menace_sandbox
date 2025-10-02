"""ROI calibration helper with lightweight drift detection."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Tuple

from .logging_utils import get_logger, log_record
from .dynamic_path_router import resolve_path

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency
    import joblib
except Exception:  # pragma: no cover - import guard
    joblib = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - import guard
    XGBRegressor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover - import guard
    Ridge = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover - import guard
    ks_2samp = None  # type: ignore


logger = None  # lazily initialized


class TruthAdapter:
    """Calibrates ROI predictions and monitors feature drift."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_type: str = "ridge",
        ridge_params: dict[str, Any] | None = None,
        xgb_params: dict[str, Any] | None = None,
        psi_threshold: float | None = None,
        ks_threshold: float | None = None,
    ) -> None:
        """Create adapter.

        Parameters
        ----------
        model_path:
            Where the trained model and metadata are persisted.
        model_type:
            ``"ridge"`` or ``"xgboost"``. If ``"xgboost"`` is requested but
            the dependency is unavailable, a ridge model is used instead.
        ridge_params:
            Optional parameters passed to :class:`~sklearn.linear_model.Ridge`.
        xgb_params:
            Optional parameters passed to :class:`xgboost.XGBRegressor`.
        psi_threshold:
            Override the Population Stability Index threshold for drift detection.
        ks_threshold:
            Override the Kolmogorov–Smirnov statistic threshold for drift detection.
        """

        if model_path is None:
            try:
                base = resolve_path("sandbox_data")
            except FileNotFoundError:
                base = resolve_path(".") / "sandbox_data"
                base.mkdir(parents=True, exist_ok=True)
            model_path = Path(base) / "truth_adapter.pkl"

        self.model_path = Path(model_path)
        self.model_type = model_type
        self.ridge_params = ridge_params or {}
        self.xgb_params = xgb_params or {}
        self.model = None
        self.metadata: dict = {}
        self._load()

        thresholds = self.metadata.get("thresholds", {})
        if psi_threshold is None:
            psi_threshold = thresholds.get("psi")
        if ks_threshold is None:
            ks_threshold = thresholds.get("ks")
        if psi_threshold is None or ks_threshold is None:
            try:  # pragma: no cover - optional dependency
                from sandbox_settings import SandboxSettings

                cfg = SandboxSettings()
                if psi_threshold is None:
                    psi_threshold = getattr(cfg, "psi_threshold", None)
                if ks_threshold is None:
                    ks_threshold = getattr(cfg, "ks_threshold", None)
            except Exception:
                pass
        self.drift_threshold = float(psi_threshold) if psi_threshold is not None else 0.25
        self.ks_threshold = float(ks_threshold) if ks_threshold is not None else 0.2
        self.metadata.setdefault("thresholds", {})
        self.metadata["thresholds"].update({"psi": self.drift_threshold, "ks": self.ks_threshold})

    # ------------------------------------------------------------------
    # Persistence helpers
    def _make_model(self):  # pragma: no cover - trivial helper
        """Instantiate the underlying regression model."""
        if self.model_type.lower() == "xgboost" and XGBRegressor is not None:
            self.metadata["model_type"] = "xgboost"
            return XGBRegressor(**self.xgb_params)

        # Fallback to ridge regression by default
        self.metadata["model_type"] = "ridge"
        return Ridge(**self.ridge_params)

    def _load(self) -> None:
        """Load model and metadata from disk if available."""
        if self.model_path.exists():
            try:
                if joblib is not None:
                    state = joblib.load(self.model_path)
                else:
                    with self.model_path.open("rb") as f:
                        state = pickle.load(f)
                self.model = state.get("model")
                self.metadata = state.get("metadata", {})
            except Exception:
                self.model = None
                self.metadata = {}
        if self.model is None:
            self.model = self._make_model()
            model_type = self.metadata.get("model_type")
            self.metadata = {
                "model_type": model_type,
                "feature_stats": None,
                "training_stats": None,
                "last_retrained": None,
                "drift_metrics": {"psi": [], "ks": []},
                "drift_flag": False,
                "retraining_required": False,
                "needs_retrain": False,  # backwards compatibility
                "last_drift_check": None,
                "version": 0,
                "samples_seen": 0,
            }
        else:
            # Backwards compatibility for older metadata keys
            if "last_retrained" not in self.metadata:
                self.metadata["last_retrained"] = self.metadata.get("last_fit")
                self.metadata.pop("last_fit", None)
            if "version" not in self.metadata:
                self.metadata["version"] = 0
            if "samples_seen" not in self.metadata:
                self.metadata["samples_seen"] = 0

    def _save(self) -> None:
        """Persist model and metadata to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure drift metrics keys exist for persistence
        self.metadata.setdefault("model_type", self.metadata.get("model_type", "ridge"))
        self.metadata.setdefault("drift_metrics", self.metadata.get("drift_metrics", {"psi": [], "ks": []}))
        self.metadata.setdefault("thresholds", {})
        self.metadata["thresholds"].update({"psi": self.drift_threshold, "ks": self.ks_threshold})
        self.metadata.setdefault("last_drift_check", None)
        self.metadata.setdefault("training_stats", None)
        self.metadata.setdefault("last_retrained", self.metadata.get("last_retrained"))
        self.metadata.setdefault("drift_flag", False)
        self.metadata.setdefault("retraining_required", False)
        self.metadata.setdefault("needs_retrain", self.metadata.get("retraining_required", False))
        self.metadata.setdefault("version", self.metadata.get("version", 0))
        self.metadata.setdefault("samples_seen", self.metadata.get("samples_seen", 0))
        state = {"model": self.model, "metadata": self.metadata}
        if joblib is not None:
            joblib.dump(state, self.model_path)
        else:
            with self.model_path.open("wb") as f:
                pickle.dump(state, f)

    # ------------------------------------------------------------------
    # Training / prediction
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        cross_validate: bool = False,
    ) -> None:
        """Train the underlying model and record feature distributions.

        Parameters
        ----------
        X, y:
            Training features and targets.
        cross_validate:
            When ``True`` and both ridge and XGBoost are available, evaluates
            both models on a hold-out split and keeps the one with the lowest
            validation error.
        """

        if (
            cross_validate
            and Ridge is not None
            and XGBRegressor is not None
        ):
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(X))
            split = int(len(X) * 0.8)
            split = min(max(1, split), len(X) - 1)
            train_idx, val_idx = idx[:split], idx[split:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            ridge_model = Ridge(**self.ridge_params)
            ridge_model.fit(X_train, y_train)
            ridge_mae = float(
                np.mean(np.abs(ridge_model.predict(X_val) - y_val))
            )

            xgb_model = XGBRegressor(**self.xgb_params)
            xgb_model.fit(X_train, y_train)
            xgb_mae = float(
                np.mean(np.abs(xgb_model.predict(X_val) - y_val))
            )

            if xgb_mae < ridge_mae:
                self.model = xgb_model
                self.model_type = "xgboost"
            else:
                self.model = ridge_model
                self.model_type = "ridge"
            self.metadata["model_type"] = self.model_type
        else:
            if self.model is None:
                self.model = self._make_model()
        self.model.fit(X, y)

        feature_stats = []
        for col in range(X.shape[1]):
            col_data = X[:, col]
            mean = float(np.mean(col_data))
            std = float(np.std(col_data) + 1e-8)
            counts, bins = np.histogram(col_data, bins=10)
            counts = counts / counts.sum() if counts.sum() else counts
            feature_stats.append({
                "mean": mean,
                "std": std,
                "bins": bins,
                "counts": counts,
            })

        preds = self.model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((preds - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot

        zeros = [0.0 for _ in feature_stats]
        self.metadata.update(
            {
                "feature_stats": feature_stats,
                "training_stats": {"mae": mae, "r2": r2},
                "last_retrained": time.time(),
                "drift_metrics": {"psi": zeros.copy(), "ks": zeros.copy()},
                "drift_flag": False,
                "retraining_required": False,
                "needs_retrain": False,
                "last_drift_check": None,
                "version": self.metadata.get("version", 0) + 1,
                "samples_seen": self.metadata.get("samples_seen", 0) + len(X),
            }
        )
        self._save()

    def partial_fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Incrementally update the model with a new batch of data."""
        if self.model is None:
            self.model = self._make_model()

        if self.metadata.get("feature_stats") is None:
            # No existing training; fall back to full fit
            self.fit(X, y)
            return

        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y)
        elif self.metadata.get("model_type") == "xgboost" and hasattr(self.model, "fit"):
            # XGBoost supports warm starting via the ``xgb_model`` parameter
            self.model.fit(X, y, xgb_model=self.model)
        else:  # fallback to refitting on new batch
            self.model.fit(X, y)

        n_prev = self.metadata.get("samples_seen", 0)
        n_new = X.shape[0]
        n_total = n_prev + n_new

        stats = self.metadata.get("feature_stats")
        if stats and len(stats) == X.shape[1]:
            updated_stats = []
            for i, fs in enumerate(stats):
                col = X[:, i]
                mean_old = fs["mean"]
                std_old = fs["std"]
                bins = np.asarray(fs["bins"])
                counts_old = np.asarray(fs["counts"]) * n_prev

                mean_new = float(np.mean(col))
                std_new = float(np.std(col) + 1e-8)
                counts_new, _ = np.histogram(col, bins=bins)
                counts_total = counts_old + counts_new

                mean_total = (mean_old * n_prev + mean_new * n_new) / n_total
                var_old = std_old ** 2
                var_new = std_new ** 2
                m2 = (
                    n_prev * (var_old + mean_old ** 2)
                    + n_new * (var_new + mean_new ** 2)
                ) / n_total
                std_total = float(np.sqrt(max(m2 - mean_total ** 2, 0.0)) + 1e-8)

                updated_stats.append(
                    {
                        "mean": mean_total,
                        "std": std_total,
                        "bins": bins,
                        "counts": counts_total / counts_total.sum()
                        if counts_total.sum()
                        else counts_total,
                    }
                )
            feature_stats = updated_stats
        else:
            feature_stats = []
            for col in range(X.shape[1]):
                col_data = X[:, col]
                mean = float(np.mean(col_data))
                std = float(np.std(col_data) + 1e-8)
                counts, bins = np.histogram(col_data, bins=10)
                counts = counts / counts.sum() if counts.sum() else counts
                feature_stats.append(
                    {"mean": mean, "std": std, "bins": bins, "counts": counts}
                )

        preds = self.model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((preds - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot

        zeros = [0.0 for _ in feature_stats]
        self.metadata.update(
            {
                "feature_stats": feature_stats,
                "training_stats": {"mae": mae, "r2": r2},
                "last_retrained": time.time(),
                "drift_metrics": {"psi": zeros.copy(), "ks": zeros.copy()},
                "drift_flag": False,
                "retraining_required": False,
                "needs_retrain": False,
                "last_drift_check": None,
                "version": self.metadata.get("version", 0) + 1,
                "samples_seen": n_total,
            }
        )
        self._save()

    def predict(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], bool]:
        """Return predictions and flag if distribution drift is detected."""
        if self.model is None:
            raise RuntimeError("Model is not trained")
        _, drift = self.check_drift(X)
        preds = self.model.predict(X)
        if drift:
            self.metadata["warning"] = "Distribution drift detected; predictions may be unreliable."
        else:
            self.metadata.pop("warning", None)
        low_conf = bool(self.metadata.get("drift_flag", False))
        # Persist warning state
        self._save()
        return preds, low_conf

    # ------------------------------------------------------------------
    # Drift detection
    def check_drift(self, X_recent: NDArray[np.float64]) -> Tuple[dict[str, list[float]], bool]:
        """Check for distribution drift using PSI and KS tests."""
        stats = self.metadata.get("feature_stats")
        if not stats:
            return {"psi": [], "ks": []}, False

        thresholds = self.metadata.get("thresholds", {})
        psi_threshold = thresholds.get("psi", self.drift_threshold)
        ks_threshold = thresholds.get("ks", self.ks_threshold)

        psi_values: list[float] = []
        ks_values: list[float] = []
        drift_detected = False
        drift_features: list[dict[str, float | int]] = []

        for i, fs in enumerate(stats):
            expected = np.array(fs["counts"])
            bins = fs["bins"]
            actual_counts, _ = np.histogram(X_recent[:, i], bins=bins)
            actual = actual_counts / actual_counts.sum() if actual_counts.sum() else actual_counts

            expected = np.where(expected == 0, 0.0001, expected)
            actual = np.where(actual == 0, 0.0001, actual)

            # Population Stability Index
            psi = np.sum((actual - expected) * np.log(actual / expected))
            psi_values.append(float(psi))

            # Kolmogorov–Smirnov statistic
            if ks_2samp is not None:
                mids = (bins[:-1] + bins[1:]) / 2
                sample_size = max(int(X_recent.shape[0]), 1000)
                exp_sample = np.repeat(mids, np.maximum((expected * sample_size).astype(int), 1))
                ks_stat, _ = ks_2samp(exp_sample, X_recent[:, i])
                ks_val = float(ks_stat)
            else:  # fallback using histogram CDFs
                exp_cdf = np.cumsum(expected)
                act_cdf = np.cumsum(actual)
                ks_val = float(np.max(np.abs(exp_cdf - act_cdf)))
            ks_values.append(ks_val)

            if psi > psi_threshold or ks_val > ks_threshold:
                drift_detected = True
                drift_features.append({"index": i, "psi": float(psi), "ks": ks_val})

        metrics = {"psi": psi_values, "ks": ks_values}

        self.metadata["drift_metrics"] = metrics
        # Keep legacy keys for compatibility
        self.metadata["psi"] = psi_values
        self.metadata["ks"] = ks_values
        self.metadata["drift_flag"] = drift_detected
        self.metadata["retraining_required"] = drift_detected
        self.metadata["needs_retrain"] = drift_detected
        self.metadata["last_drift_check"] = time.time()
        if drift_detected:
            global logger
            if logger is None:
                logger = get_logger(__name__)
            logger.warning(
                "feature drift detected",
                extra=log_record(drift_features=drift_features),
            )
        self._save()
        return metrics, drift_detected

    # ------------------------------------------------------------------
    # Reset / retrain helpers
    def reset(self) -> None:
        """Clear model state and metadata for a fresh retrain."""
        self.model = self._make_model()
        thresholds = self.metadata.get(
            "thresholds", {"psi": self.drift_threshold, "ks": self.ks_threshold}
        )
        self.metadata = {
            "model_type": self.metadata.get("model_type", self.model_type),
            "feature_stats": None,
            "training_stats": None,
            "last_retrained": None,
            "drift_metrics": {"psi": [], "ks": []},
            "drift_flag": False,
            "retraining_required": False,
            "needs_retrain": False,
            "last_drift_check": None,
            "version": 0,
            "samples_seen": 0,
            "thresholds": thresholds,
        }
        self._save()

    def retrain(self, X: NDArray[np.float64], y: NDArray[np.float64], *, cross_validate: bool = False) -> None:
        """Reset the model and fit from scratch on ``X`` and ``y``."""
        self.reset()
        self.fit(X, y, cross_validate=cross_validate)
