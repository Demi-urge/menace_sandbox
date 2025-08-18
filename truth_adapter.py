"""ROI calibration helper with lightweight drift detection."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Tuple

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


class TruthAdapter:
    """Calibrates ROI predictions and monitors feature drift."""

    def __init__(
        self,
        model_path: str | Path = "sandbox_data/truth_adapter.pkl",
        use_xgboost: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.use_xgboost = use_xgboost
        self.model = None
        self.metadata: dict = {}
        self.drift_threshold = 0.25  # PSI threshold for drift
        self.ks_threshold = 0.2  # KS statistic threshold
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _make_model(self):  # pragma: no cover - trivial helper
        """Instantiate the underlying regression model."""
        if self.use_xgboost and XGBRegressor is not None:
            return XGBRegressor()
        return Ridge()

    def _load_state(self) -> None:
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
            self.metadata = {
                "feature_stats": None,
                "last_fit": None,
                "psi": None,
                "ks": None,
                "drift_flag": False,
                "needs_retrain": False,
                "last_drift_check": None,
            }

    def _save_state(self) -> None:
        """Persist model and metadata to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure drift metrics keys exist for persistence
        self.metadata.setdefault("psi", None)
        self.metadata.setdefault("ks", None)
        self.metadata.setdefault("last_drift_check", None)
        state = {"model": self.model, "metadata": self.metadata}
        if joblib is not None:
            joblib.dump(state, self.model_path)
        else:
            with self.model_path.open("wb") as f:
                pickle.dump(state, f)

    # ------------------------------------------------------------------
    # Training / prediction
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Train the underlying model and record feature distributions."""
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

        self.metadata.update(
            {
                "feature_stats": feature_stats,
                "last_fit": time.time(),
                "psi": [0.0 for _ in feature_stats],
                "ks": [0.0 for _ in feature_stats],
                "drift_flag": False,
                "needs_retrain": False,
                "last_drift_check": None,
            }
        )
        self._save_state()

    def predict(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], bool]:
        """Return predictions and flag if distribution drift is detected."""
        if self.model is None:
            raise RuntimeError("Model is not trained")
        drift = self.check_drift(X)
        preds = self.model.predict(X)
        if drift:
            self.metadata["warning"] = "Distribution drift detected; predictions may be unreliable."
        else:
            self.metadata.pop("warning", None)
        low_conf = bool(self.metadata.get("drift_flag", False))
        # Persist warning state
        self._save_state()
        return preds, low_conf

    # ------------------------------------------------------------------
    # Drift detection
    def check_drift(self, X_recent: NDArray[np.float64]) -> bool:
        """Check for distribution drift using PSI and KS tests."""
        stats = self.metadata.get("feature_stats")
        if not stats:
            return False

        psi_values: list[float] = []
        ks_values: list[float] = []
        drift_detected = False

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

            if psi > self.drift_threshold or ks_val > self.ks_threshold:
                drift_detected = True

        self.metadata["psi"] = psi_values
        self.metadata["ks"] = ks_values
        self.metadata["drift_flag"] = drift_detected
        self.metadata["needs_retrain"] = drift_detected
        self.metadata["last_drift_check"] = time.time()
        self._save_state()
        return drift_detected
