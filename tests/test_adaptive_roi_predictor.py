"""Tests for :mod:`adaptive_roi_predictor` and ROITracker integration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import json

import numpy as np
import pytest

from menace_sandbox.adaptive_roi_dataset import build_dataset
from menace_sandbox.adaptive_roi_predictor import AdaptiveROIPredictor
from menace_sandbox.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
from menace_sandbox.evolution_history_db import EvolutionHistoryDB
from menace_sandbox.roi_tracker import ROITracker


def test_build_dataset(tmp_path: Path) -> None:
    """Dataset generation combines evolution, ROI and evaluation data."""

    evo_path = tmp_path / "evolution_history.db"
    eval_path = tmp_path / "evaluation_history.db"
    roi_path = tmp_path / "roi.db"

    evo = EvolutionHistoryDB(evo_path)
    eval_db = EvaluationHistoryDB(eval_path)
    conn = sqlite3.connect(roi_path)
    conn.execute(
        "CREATE TABLE action_roi(action TEXT, revenue REAL, api_cost REAL, cpu_seconds REAL, success_rate REAL, ts TEXT)"
    )

    # ROI records before and after the evolution event
    conn.execute(
        "INSERT INTO action_roi VALUES (?,?,?,?,?,?)",
        ("test", 10.0, 1.0, 0.0, 1.0, "2022-12-31T00:00:00"),
    )
    conn.execute(
        "INSERT INTO action_roi VALUES (?,?,?,?,?,?)",
        ("test", 12.0, 1.0, 0.0, 1.0, "2023-01-02T00:00:00"),
    )
    conn.commit()

    # Evaluation score after the evolution event
    eval_db.add(
        EvaluationRecord(
            engine="test", cv_score=0.5, passed=True, ts="2023-01-01T01:00:00"
        )
    )

    # Evolution event
    evo.conn.execute(
        "INSERT INTO evolution_history(action, before_metric, after_metric, roi, ts) VALUES (?,?,?,?,?)",
        ("test", 1.0, 2.0, 0.0, "2023-01-01T00:00:00"),
    )
    evo.conn.commit()

    X, y, g = build_dataset(evo_path, roi_path, eval_path)
    tracker = ROITracker()
    base_metrics = set(tracker.metrics_history) | set(tracker.synergy_metrics_history)
    n_features = 6 + len(base_metrics) + 5 + 4
    assert X.shape == (1, n_features)
    assert y.shape == (1,)
    # Target is revenue minus API cost after the event
    assert y[0] == pytest.approx(11.0)
    assert g.tolist() == ["linear"]


def test_classifier_training_and_prediction(monkeypatch):
    """Model trains classifier and uses it for growth prediction."""

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    g = np.array(
        ["marginal", "marginal", "linear", "linear", "exponential", "exponential"],
        dtype=object,
    )

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", lambda: (X, y, g)
    )
    predictor = AdaptiveROIPredictor()
    assert predictor._model is not None
    if predictor._classifier is None:
        pytest.skip("classifier not available")

    roi_seq, growth, conf = predictor.predict(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], horizon=6
    )
    assert roi_seq[-1] == pytest.approx(5.0, abs=1e-3)
    assert growth == "exponential"
    assert len(conf) == len(roi_seq)
    assert (
        predictor.predict_growth_type(
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], horizon=6
        )
        == "exponential"
    )


def test_cross_validation_persistence(tmp_path, monkeypatch):
    """Cross-validation stores metadata about the best model."""

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    g = np.array(["marginal", "linear", "linear", "linear"], dtype=object)
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", lambda: (X, y, g)
    )
    model_path = tmp_path / "adaptive_roi.pkl"
    predictor = AdaptiveROIPredictor(model_path=model_path, cv=2)
    assert predictor.best_params is not None
    assert predictor.best_score is not None
    meta_path = model_path.with_suffix(".meta.json")
    assert meta_path.exists()


def test_selected_features_saved(tmp_path, monkeypatch):
    X = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    g = np.array(["linear", "linear"], dtype=object)
    names = ["a", "b", "c"]

    def fake_build_dataset(*_args, **_kwargs):
        return X, y, g, names

    class DummyModel:
        def fit(self, X, y):
            self.feature_importances_ = np.array([0.1, 0.6, 0.3])
            return self

        def predict(self, X):  # pragma: no cover - trivial
            return np.zeros(len(X))

        def get_params(self):  # pragma: no cover - trivial
            return {}

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build_dataset
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.GradientBoostingRegressor",
        DummyModel,
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.SGDRegressor", None
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.LinearRegression", None
    )

    model_path = tmp_path / "model.pkl"
    predictor = AdaptiveROIPredictor(model_path=model_path, cv=0, param_grid={})
    meta_path = model_path.with_suffix(".meta.json")
    data = json.loads(meta_path.read_text())
    assert data["selected_features"][:3] == ["b", "c", "a"]


def test_evaluate_model_retrains(monkeypatch, tmp_path):
    """evaluate_model triggers retraining when error is high."""

    tracker = ROITracker()

    db_path = tmp_path / "roi_events.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE roi_prediction_events (
            predicted_roi REAL,
            actual_roi REAL,
            predicted_class TEXT,
            actual_class TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO roi_prediction_events (predicted_roi, actual_roi, predicted_class, actual_class) VALUES (?,?,?,?)",
        (0.0, 1.0, "linear", "exponential"),
    )
    conn.execute(
        "INSERT INTO roi_prediction_events (predicted_roi, actual_roi, predicted_class, actual_class) VALUES (?,?,?,?)",
        (0.0, 1.0, "linear", "exponential"),
    )
    conn.commit()
    conn.close()

    called: list[int] = []

    class DummyPredictor:
        def train(self):
            called.append(1)

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.AdaptiveROIPredictor",
        DummyPredictor,
    )

    acc, mae = tracker.evaluate_model(
        window=2,
        mae_threshold=0.1,
        acc_threshold=0.9,
        roi_events_path=str(db_path),
    )
    assert mae > 0.1
    assert acc < 0.9
    assert called  # retraining triggered


def test_tracker_integration(monkeypatch):
    """ROITracker uses predictor to log class predictions and evaluate."""

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset",
        lambda: (np.array([[0.0]]), np.array([0.0]), np.array(["linear"], dtype=object)),
    )
    predictor = AdaptiveROIPredictor()
    tracker = ROITracker()
    tracker._adaptive_predictor = predictor

    called: list[int] = []

    def fake_eval(self, **_kwargs):
        called.append(1)
        return (1.0, 0.0)

    monkeypatch.setattr(ROITracker, "evaluate_model", fake_eval)

    tracker.roi_history = [0.1, 0.2]
    tracker._next_prediction = 0.25
    tracker._next_category = "linear"
    tracker.update(0.0, 0.3)

    assert tracker.predicted_classes == ["linear"]
    assert len(tracker.actual_classes) == 1
    assert called  # evaluate_model invoked


def test_build_dataset_missing_database(tmp_path: Path) -> None:
    """build_dataset raises when required tables are absent."""

    evo_path = tmp_path / "evolution_history.db"
    eval_path = tmp_path / "evaluation_history.db"
    EvolutionHistoryDB(evo_path)
    EvaluationHistoryDB(eval_path)
    roi_path = tmp_path / "roi.db"  # no tables created

    with pytest.raises(sqlite3.OperationalError):
        build_dataset(evo_path, roi_path, eval_path)


def test_corrupted_model_file(tmp_path: Path, monkeypatch) -> None:
    """Corrupted model files are ignored and retraining proceeds."""

    model_path = tmp_path / "adaptive_roi.pkl"
    model_path.write_text("not a pickle")

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    g = np.array(["marginal", "linear", "linear", "linear"], dtype=object)
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", lambda: (X, y, g)
    )

    predictor = AdaptiveROIPredictor(model_path=model_path, cv=0, param_grid={})
    assert predictor._model is not None
    roi_seq, growth, conf = predictor.predict([[1.0], [2.0], [3.0]], horizon=3)
    assert roi_seq[-1] == pytest.approx(3.0, abs=1e-3)
    assert growth in {"marginal", "linear", "exponential"}
    assert len(conf) == len(roi_seq)

