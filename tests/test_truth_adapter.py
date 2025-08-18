"""Regression and drift behaviour for :class:`TruthAdapter`."""

from __future__ import annotations

import sys
import types
from pathlib import Path
import logging

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

    metrics, drift_flag = adapter.check_drift(X_shifted)
    assert drift_flag is True
    assert set(metrics) == {"psi", "ks"}
    # check_drift should raise flags for downstream callers
    assert adapter.metadata["retraining_required"] is True

    preds, low_conf = adapter.predict(X_shifted)
    assert low_conf is True
    assert adapter.metadata["drift_flag"] is True
    # Predict should surface a warning message when drift is present
    assert "warning" in adapter.metadata


def test_thresholds_from_config(monkeypatch, tmp_path: Path) -> None:
    ss_mod = types.ModuleType("sandbox_settings")
    ss_mod.SandboxSettings = lambda: types.SimpleNamespace(
        psi_threshold=0.1, ks_threshold=0.05
    )
    monkeypatch.setitem(sys.modules, "sandbox_settings", ss_mod)

    adapter = TruthAdapter(tmp_path / "truth.pkl")
    assert adapter.drift_threshold == 0.1
    assert adapter.ks_threshold == 0.05
    assert adapter.metadata["thresholds"] == {"psi": 0.1, "ks": 0.05}


def test_drift_logging(tmp_path: Path, caplog) -> None:
    X, y = _make_data()
    adapter = TruthAdapter(tmp_path / "truth.pkl")
    adapter.fit(X, y)
    X_shifted = X + 5.0
    with caplog.at_level(logging.WARNING):
        adapter.check_drift(X_shifted)
    rec = next(r for r in caplog.records if "feature drift detected" in r.message)
    assert hasattr(rec, "drift_features")
    assert rec.drift_features


def test_partial_fit_updates_metadata(tmp_path: Path) -> None:
    X, y = _make_data(200)
    X1, y1 = X[:100], y[:100]
    X2, y2 = X[100:], y[100:]
    adapter = TruthAdapter(tmp_path / "truth.pkl")
    adapter.fit(X1, y1)
    v1 = adapter.metadata["version"]
    n1 = adapter.metadata["samples_seen"]
    preds_before, _ = adapter.predict(X2)
    err_before = float(np.mean(np.abs(preds_before - y2)))
    adapter.partial_fit(X2, y2)
    assert adapter.metadata["version"] == v1 + 1
    assert adapter.metadata["samples_seen"] == n1 + len(X2)
    preds_after, _ = adapter.predict(X2)
    err_after = float(np.mean(np.abs(preds_after - y2)))
    assert err_after <= err_before + 1e-3


def test_reset_clears_state(tmp_path: Path) -> None:
    X, y = _make_data(50)
    adapter = TruthAdapter(tmp_path / "truth.pkl")
    adapter.fit(X, y)
    adapter.reset()
    assert adapter.metadata["version"] == 0
    assert adapter.metadata["samples_seen"] == 0
    assert adapter.metadata["feature_stats"] is None

