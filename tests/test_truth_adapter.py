"""Regression and drift behaviour for :class:`TruthAdapter`."""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
import logging

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# TruthAdapter optionally uses xgboost; stub the module so the tests avoid
# pulling in the heavy dependency.
xgb_stub = types.ModuleType("xgboost")
xgb_stub.XGBRegressor = None
sys.modules.setdefault("xgboost", xgb_stub)

from menace.truth_adapter import TruthAdapter


class DummyXGBRegressor:
    """Minimal stand-in for :class:`xgboost.XGBRegressor` used in tests."""

    def __init__(self, **kwargs):
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        return X @ self.coef_


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


def test_metadata_thresholds_skip_settings(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class SandboxSettings:
        def __init__(self):
            calls.append("called")

    monkeypatch.setitem(
        sys.modules,
        "sandbox_settings",
        types.SimpleNamespace(SandboxSettings=SandboxSettings),
    )

    model_path = tmp_path / "truth.pkl"
    state = {"model": "stub", "metadata": {"thresholds": {"psi": 0.15, "ks": 0.05}}}
    with model_path.open("wb") as fh:
        pickle.dump(state, fh)

    adapter = TruthAdapter(model_path)
    assert adapter.drift_threshold == 0.15
    assert adapter.ks_threshold == 0.05
    assert calls == []


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


def test_scipy_ks_drift_detection(monkeypatch, tmp_path: Path) -> None:
    """When SciPy is available, the KS statistic should come from ``ks_2samp``."""

    def fake_ks_2samp(a, b):
        return 0.5, 0.0

    stats_mod = types.SimpleNamespace(ks_2samp=fake_ks_2samp)
    scipy_mod = types.SimpleNamespace(stats=stats_mod)
    monkeypatch.setitem(sys.modules, "scipy", scipy_mod)
    monkeypatch.setitem(sys.modules, "scipy.stats", stats_mod)
    monkeypatch.setattr("menace.truth_adapter.ks_2samp", fake_ks_2samp, raising=False)

    X, y = _make_data()
    adapter = TruthAdapter(tmp_path / "truth.pkl")
    adapter.fit(X, y)
    metrics, drift_flag = adapter.check_drift(X + 5.0)
    assert drift_flag is True
    assert metrics["ks"][0] == 0.5


def test_histogram_ks_fallback(monkeypatch, tmp_path: Path) -> None:
    """If SciPy is missing, histogram-based KS approximation is used."""

    monkeypatch.setattr("menace.truth_adapter.ks_2samp", None, raising=False)
    X, y = _make_data()
    adapter = TruthAdapter(tmp_path / "truth.pkl")
    adapter.fit(X, y)
    X_shifted = X + 5.0
    metrics, drift_flag = adapter.check_drift(X_shifted)
    assert drift_flag is True
    fs0 = adapter.metadata["feature_stats"][0]
    expected = np.array(fs0["counts"])
    bins = fs0["bins"]
    actual_counts, _ = np.histogram(X_shifted[:, 0], bins=bins)
    actual = actual_counts / actual_counts.sum() if actual_counts.sum() else actual_counts
    expected = np.where(expected == 0, 0.0001, expected)
    actual = np.where(actual == 0, 0.0001, actual)
    exp_cdf = np.cumsum(expected)
    act_cdf = np.cumsum(actual)
    ks_expected = float(np.max(np.abs(exp_cdf - act_cdf)))
    assert metrics["ks"][0] == pytest.approx(ks_expected)


def test_xgboost_model_selection(monkeypatch, tmp_path: Path) -> None:
    """Cross-validation should pick XGBoost when it performs better."""

    class BadRidge:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr("menace.truth_adapter.XGBRegressor", DummyXGBRegressor, raising=False)
    monkeypatch.setattr("menace.truth_adapter.Ridge", BadRidge, raising=False)

    X, y = _make_data()
    adapter = TruthAdapter(tmp_path / "truth.pkl")
    adapter.fit(X, y, cross_validate=True)
    assert adapter.metadata["model_type"] == "xgboost"
    preds, low_conf = adapter.predict(X)
    assert not low_conf
    assert np.allclose(preds, y, atol=0.1)

    adapter2 = TruthAdapter(tmp_path / "truth.pkl")
    assert adapter2.metadata["model_type"] == "xgboost"
    preds2, low_conf2 = adapter2.predict(X)
    assert not low_conf2
    assert np.allclose(preds, preds2)


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

