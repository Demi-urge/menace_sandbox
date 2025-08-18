"""Unit tests for TruthAdapter calibration and drift detection."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Avoid importing heavy optional dependency xgboost in the test environment.
# TruthAdapter falls back to ``sklearn.linear_model.Ridge`` when XGBoost is not
# available.  Stubbing the module keeps the test lightweight and deterministic.
xgb_stub = types.ModuleType("xgboost")
xgb_stub.XGBRegressor = None
sys.modules.setdefault("xgboost", xgb_stub)

from menace.truth_adapter import TruthAdapter


def _synth_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple linear data for calibration tests."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 3))
    coef = np.array([1.5, -2.0, 0.5])
    y = X @ coef  # perfect linear relationship
    return X, y


def test_fit_predict_and_reload(tmp_path: Path) -> None:
    """Model should learn mapping and persist state to disk."""

    X, y = _synth_data()
    model_path = tmp_path / "truth.pkl"

    ta = TruthAdapter(model_path)
    ta.fit(X, y)

    preds, low_conf = ta.predict(X)
    assert not low_conf
    # The linear relationship should be recovered with low error.
    assert np.max(np.abs(preds - y)) < 0.1
    assert model_path.exists()

    # Reload adapter from disk and ensure predictions are identical.
    ta_reloaded = TruthAdapter(model_path)
    preds2, low_conf2 = ta_reloaded.predict(X)
    assert not low_conf2
    assert np.allclose(preds, preds2)
    # Metadata such as last_fit should persist across reloads.
    assert ta_reloaded.metadata["last_fit"] == ta.metadata["last_fit"]
    assert ta_reloaded.metadata["feature_stats"] is not None


def test_drift_detection_sets_low_confidence(tmp_path: Path) -> None:
    """Shifted feature distribution should trigger drift and low-confidence."""

    X, y = _synth_data()
    model_path = tmp_path / "truth.pkl"

    ta = TruthAdapter(model_path)
    ta.fit(X, y)

    # Create a strongly shifted distribution.
    X_shifted = X + 10.0

    assert ta.check_drift(X_shifted) is True
    preds, low_conf = ta.predict(X_shifted)
    assert low_conf is True
    assert ta.metadata["drift_flag"] is True
    # PSI values should be recorded for each feature.
    assert isinstance(ta.metadata["psi"], list)
    assert len(ta.metadata["psi"]) == X.shape[1]
    # KS statistics and timestamps should also persist.
    assert isinstance(ta.metadata["ks"], list)
    assert len(ta.metadata["ks"]) == X.shape[1]
    assert ta.metadata["last_drift_check"] is not None
    assert "warning" in ta.metadata
