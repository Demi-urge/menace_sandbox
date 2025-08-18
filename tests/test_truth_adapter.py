"""Regression and drift behaviour for :class:`TruthAdapter`."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# TruthAdapter optionally uses xgboost; stub the module so the tests avoid
# pulling in the heavy dependency.
xgb_stub = types.ModuleType("xgboost")
xgb_stub.XGBRegressor = None
sys.modules.setdefault("xgboost", xgb_stub)

from menace.truth_adapter import TruthAdapter


def _make_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple linear relationship for calibration."""

    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    coef = np.array([1.5, -2.0, 0.5])
    y = X @ coef
    return X, y


def test_train_predict_reload(tmp_path: Path) -> None:
    """Training should learn mapping and persist identical predictions."""

    X, y = _make_data()
    model_path = tmp_path / "truth.pkl"

    adapter = TruthAdapter(model_path)
    adapter.fit(X, y)

    preds, low_conf = adapter.predict(X)
    assert not low_conf
    assert np.allclose(preds, y, atol=0.1)

    adapter_reloaded = TruthAdapter(model_path)
    preds2, low_conf2 = adapter_reloaded.predict(X)
    assert not low_conf2
    assert np.allclose(preds, preds2)


def test_shifted_features_trigger_drift(tmp_path: Path) -> None:
    """Shifted feature distribution should raise drift flag and low-confidence."""

    X, y = _make_data()
    model_path = tmp_path / "truth.pkl"

    adapter = TruthAdapter(model_path)
    adapter.fit(X, y)

    X_shifted = X + 5.0

    assert adapter.check_drift(X_shifted) is True
    preds, low_conf = adapter.predict(X_shifted)
    assert low_conf is True
    assert adapter.metadata["drift_flag"] is True

