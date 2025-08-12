"""Tests for :mod:`adaptive_roi_predictor` and ROITracker integration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
    n_features = 6 + len(ROITracker().metrics_history)
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

    roi, growth = predictor.predict([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    assert roi == pytest.approx(5.0, abs=1e-3)
    assert growth == "exponential"


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


def test_evaluate_model_retrains(monkeypatch):
    """evaluate_model triggers CLI retraining when error is high."""

    tracker = ROITracker()
    tracker.predicted_roi = [0.0, 0.0]
    tracker.actual_roi = [1.0, 1.0]
    tracker.predicted_classes = ["linear", "linear"]
    tracker.actual_classes = ["exponential", "exponential"]

    calls: list[list[str]] = []

    def fake_popen(cmd, **_kwargs):
        calls.append(cmd)  # type: ignore[arg-type]
        class Dummy:
            pass
        return Dummy()

    monkeypatch.setattr(
        "menace_sandbox.roi_tracker.subprocess.Popen", fake_popen
    )

    acc, mae = tracker.evaluate_model(
        window=2, mae_threshold=0.1, acc_threshold=0.9
    )
    assert mae > 0.1
    assert acc < 0.9
    assert calls  # retraining triggered


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

