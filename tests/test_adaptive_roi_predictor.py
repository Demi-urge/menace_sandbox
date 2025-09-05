"""Tests for :mod:`adaptive_roi_predictor` and ROITracker integration."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import sqlite3
import sys
import types

import db_router
from menace_sandbox.adaptive_roi_dataset import build_dataset
from menace_sandbox import adaptive_roi_predictor as arp_mod
from menace_sandbox.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
from menace_sandbox.evolution_history_db import EvolutionHistoryDB
from menace_sandbox import roi_tracker as roi_mod
from menace_sandbox import telemetry_backend as tb_mod


def test_build_dataset(tmp_path: Path, monkeypatch) -> None:
    """Dataset generation combines evolution, ROI and evaluation data."""

    evo_path = tmp_path / "evolution_history.db"
    router = db_router.DBRouter(
        "pred", str(tmp_path / "pred.sqlite"), str(tmp_path / "pred.sqlite")
    )

    evo = EvolutionHistoryDB(evo_path)
    eval_db = EvaluationHistoryDB(router=router)
    conn = router.get_connection("action_roi")
    conn.execute(
        "CREATE TABLE action_roi(action TEXT, revenue REAL, api_cost REAL, cpu_seconds REAL, success_rate REAL, ts TEXT)"
    )

    class DummyEmb:
        def __init__(self, *args, **kwargs):
            pass

        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(OpenAIEmbeddings=DummyEmb),
    )
    eval_db.conn.execute(
        "ALTER TABLE evaluation_history ADD COLUMN gpt_feedback TEXT"
    )

    # ROI records before and after the evolution event
    conn.execute(
        "INSERT INTO action_roi VALUES (?,?,?,?,?,?)",
        ("test", 10.0, 1.0, 0.0, 1.0, "2022-12-31T00:00:00"),
    )
    for i, rev in enumerate([12.0, 14.0, 16.0, 18.0, 20.0], start=1):
        conn.execute(
            "INSERT INTO action_roi VALUES (?,?,?,?,?,?)",
            (
                "test",
                rev,
                1.0,
                0.0,
                1.0,
                f"2023-01-{i+1:02d}T00:00:00",
            ),
        )
    conn.commit()

    eval_db.conn.execute(
        "INSERT INTO evaluation_history(engine, cv_score, passed, error, ts, gpt_feedback) VALUES (?,?,?,?,?,?)",
        ("test", 0.5, 1, "", "2023-01-01T01:00:00", "ok"),
    )
    eval_db.conn.commit()

    evo.conn.execute(
        "INSERT INTO evolution_history(action, before_metric, after_metric, roi, ts) VALUES (?,?,?,?,?)",
        ("test", 1.0, 2.0, 0.0, "2023-01-01T00:00:00"),
    )
    evo.conn.commit()

    X, y, g, names = build_dataset(
        evo_path, router=router, return_feature_names=True
    )
    assert X.shape == (1, len(names))
    assert any(n.startswith("gpt_feedback_emb_") for n in names)
    assert y.shape == (1, 4)
    assert y.tolist()[0][:3] == [11.0, 15.0, 19.0]
    assert g.tolist() == ["linear"]


def test_classifier_training_and_prediction(monkeypatch):
    """Model trains classifier and uses it for growth prediction."""

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=float)
    y = np.column_stack(
        [np.arange(6, dtype=float), np.arange(0, 12, 2, dtype=float)]
    )
    g = np.array(
        ["marginal", "marginal", "linear", "linear", "exponential", "exponential"],
        dtype=object,
    )

    def fake_build(*_args, **kwargs):
        if kwargs.get("return_feature_names"):
            return X, y, g, []
        return X, y, g

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.GradientBoostingRegressor", None
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.SGDRegressor", None
    )
    predictor = arp_mod.AdaptiveROIPredictor(cv=0, param_grid={})
    assert predictor._model is not None
    if predictor._classifier is None:
        pytest.skip("classifier not available")

    roi_seq, growth, conf, cls_conf = predictor.predict(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], horizon=6
    )
    assert roi_seq[-1][0] == pytest.approx(5.0, abs=1e-3)
    assert roi_seq[-1][1] == pytest.approx(10.0, abs=1e-3)
    assert growth == "exponential"
    assert isinstance(cls_conf, float) or cls_conf is None
    assert len(conf) == len(roi_seq)
    assert len(conf[0]) == len(roi_seq[0])
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
    def fake_build_cv(*_args, **kwargs):
        if kwargs.get("return_feature_names"):
            return X, y, g, []
        return X, y, g

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build_cv
    )
    model_path = tmp_path / "adaptive_roi.pkl"
    predictor = arp_mod.AdaptiveROIPredictor(model_path=model_path, cv=2)
    assert predictor.best_params is not None
    assert predictor.best_score is not None
    meta_path = model_path.with_suffix(".meta.json")
    assert meta_path.exists()


def test_threshold_auto_calibration(tmp_path, monkeypatch):
    """Training derives slope/curvature thresholds when not provided."""

    X = np.zeros((5, 1), dtype=float)
    y = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 3.0, 6.0],
            [0.0, 2.0, 3.0, 3.5],
            [0.0, 0.5, 1.5, 3.0],
            [0.0, 1.5, 1.75, 1.875],
        ],
        dtype=float,
    )
    g = np.array(["linear"] * 5, dtype=object)

    def fake_build(*_args, **kwargs):
        if kwargs.get("return_feature_names"):
            return X, y, g, []
        return X, y, g

    class DummyModel:
        def __init__(self, **_kwargs):
            self._dim = 1

        def fit(self, X, y):
            self._dim = y.shape[1] if y.ndim > 1 else 1
            return self

        def predict(self, X):  # pragma: no cover - trivial
            return np.zeros((len(X), self._dim))

        def get_params(self):  # pragma: no cover - trivial
            return {}

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.GradientBoostingRegressor",
        None,
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.SGDRegressor", None
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.MultiOutputRegressor", None
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.LinearRegression", DummyModel
    )

    model_path = tmp_path / "model.pkl"
    predictor = arp_mod.AdaptiveROIPredictor(model_path=model_path, cv=0, param_grid={})

    assert predictor.slope_threshold == pytest.approx(1.5)
    assert predictor.curvature_threshold == pytest.approx(1.0)
    meta = json.loads(model_path.with_suffix(".meta.json").read_text())
    assert meta["slope_threshold"] == pytest.approx(1.5)
    assert meta["curvature_threshold"] == pytest.approx(1.0)


def test_selected_features_saved(tmp_path, monkeypatch):
    X = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    g = np.array(["linear", "linear"], dtype=object)
    names = ["a", "b", "c"]

    def fake_build_dataset(*_args, **_kwargs):
        return X, y, g, names

    class DummyModel:
        def __init__(self, **_kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):  # pragma: no cover - trivial
            return np.zeros(len(X))
        def get_params(self):  # pragma: no cover - trivial
            return {}

    class DummySelector:
        def __init__(self, score_func=None, k=10):
            self.k = k
        def fit(self, X, y):
            self.scores_ = np.array([0.1, 0.6, 0.3])
            return self
        def get_support(self, indices=False):
            order = np.argsort(self.scores_)[::-1][: self.k]
            if indices:
                return order
            mask = np.zeros_like(self.scores_, dtype=bool)
            mask[order] = True
            return mask

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build_dataset
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.SelectKBest", DummySelector
    )
    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.f_regression", lambda X, y: None
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
    predictor = arp_mod.AdaptiveROIPredictor(
        model_path=model_path,
        cv=0,
        param_grid={"GradientBoostingRegressor": {}},
    )
    assert predictor.selected_features == ["b", "c", "a"]


def test_evaluate_model_retrains(monkeypatch, tmp_path):
    """evaluate_model triggers retraining when error is high."""

    tracker = roi_mod.ROITracker()

    db_path = tmp_path / "roi_events.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE roi_prediction_events (
            predicted_roi REAL,
            actual_roi REAL,
            predicted_class TEXT,
            actual_class TEXT,
            confidence REAL,
            workflow_id TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO roi_prediction_events (predicted_roi, actual_roi, predicted_class, actual_class, confidence, workflow_id) VALUES (?,?,?,?,?,?)",
        (0.0, 1.0, "linear", "exponential", 0.5, "wf"),
    )
    conn.execute(
        "INSERT INTO roi_prediction_events (predicted_roi, actual_roi, predicted_class, actual_class, confidence, workflow_id) VALUES (?,?,?,?,?,?)",
        (0.0, 1.0, "linear", "exponential", 0.4, "wf"),
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

    db_router.init_db_router(
        "roi_tracker", local_db_path=str(db_path), shared_db_path=str(tmp_path / "shared.db")
    )
    roi_mod.router = db_router.GLOBAL_ROUTER

    acc, mae = tracker.evaluate_model(
        window=2,
        mae_threshold=0.1,
        acc_threshold=0.9,
        roi_events_path=str(db_path),
    )
    assert mae > 0.1
    assert acc < 0.9
    assert called  # retraining triggered


def test_evaluate_model_drift_retrains(monkeypatch, tmp_path):
    """Drift detection triggers retraining even when error thresholds pass."""

    tracker = roi_mod.ROITracker()

    db_path = tmp_path / "roi_events.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE roi_prediction_events (
            predicted_roi REAL,
            actual_roi REAL,
            predicted_class TEXT,
            actual_class TEXT,
            confidence REAL,
            workflow_id TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    for _ in range(5):
        conn.execute(
            "INSERT INTO roi_prediction_events (predicted_roi, actual_roi, predicted_class, actual_class, confidence, workflow_id) VALUES (?,?,?,?,?,?)",
            (0.0, 0.0, "linear", "linear", 0.1, "wf"),
        )
    for _ in range(5):
        conn.execute(
            "INSERT INTO roi_prediction_events (predicted_roi, actual_roi, predicted_class, actual_class, confidence, workflow_id) VALUES (?,?,?,?,?,?)",
            (0.0, 1.0, "linear", "linear", 0.9, "wf"),
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

    db_router.init_db_router(
        "roi_tracker2", local_db_path=str(db_path), shared_db_path=str(tmp_path / "shared.db")
    )
    roi_mod.router = db_router.GLOBAL_ROUTER

    acc, mae = tracker.evaluate_model(
        window=10,
        mae_threshold=1.0,
        acc_threshold=0.5,
        roi_events_path=str(db_path),
        drift_threshold=0.3,
    )
    assert mae < 1.0  # overall MAE within threshold
    assert acc >= 0.5  # accuracy within threshold
    assert tracker.drift_flags[-1] is True
    assert called  # retraining triggered due to drift


def test_tracker_integration(monkeypatch):
    """ROITracker uses predictor to log class predictions and evaluate."""

    def fake_build_tracker(*_args, **kwargs):
        X = np.array([[0.0]])
        y = np.array([0.0])
        g = np.array(["linear"], dtype=object)
        if kwargs.get("return_feature_names"):
            return X, y, g, []
        return X, y, g

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build_tracker
    )
    predictor = arp_mod.AdaptiveROIPredictor()
    tracker = roi_mod.ROITracker()
    tracker._adaptive_predictor = predictor

    called: list[int] = []

    def fake_eval(self, **_kwargs):
        called.append(1)
        return (1.0, 0.0)

    monkeypatch.setattr(roi_mod.ROITracker, "evaluate_model", fake_eval)

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
    EvolutionHistoryDB(evo_path)
    router = db_router.DBRouter(
        "missing", str(tmp_path / "missing.sqlite"), str(tmp_path / "missing.sqlite")
    )
    EvaluationHistoryDB(router=router)
    # no action_roi table created

    with pytest.raises(sqlite3.OperationalError):
        build_dataset(evo_path, router=router)


def test_corrupted_model_file(tmp_path: Path, monkeypatch) -> None:
    """Corrupted model files are ignored and retraining proceeds."""

    model_path = tmp_path / "adaptive_roi.pkl"
    model_path.write_text("not a pickle")

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    g = np.array(["marginal", "linear", "linear", "linear"], dtype=object)
    def fake_build_corrupt(*_args, **kwargs):
        if kwargs.get("return_feature_names"):
            return X, y, g, []
        return X, y, g

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build_corrupt
    )

    predictor = arp_mod.AdaptiveROIPredictor(model_path=model_path, cv=0, param_grid={})
    assert predictor._model is not None
    roi_seq, growth, conf, _ = predictor.predict([[1.0], [2.0], [3.0]], horizon=3)
    assert roi_seq[-1][0] == pytest.approx(3.0, abs=1e-3)
    assert growth in {"marginal", "linear", "exponential"}
    assert len(conf) == len(roi_seq)


def test_prediction_confidence_persisted_and_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_router.init_db_router(
        "roi", local_db_path=str(tmp_path / "roi_events.db"), shared_db_path=str(tmp_path / "shared.db")
    )
    importlib.reload(arp_mod)
    importlib.reload(tb_mod)
    importlib.reload(roi_mod)
    tb_mod._init_roi_events_db()
    tracker = roi_mod.ROITracker()
    tracker.roi_history = [0.1]
    tracker.record_prediction(
        0.2,
        0.3,
        predicted_class="linear",
        actual_class="linear",
        confidence=0.77,
    )
    EvaluationHistoryDB(router=db_router.GLOBAL_ROUTER)
    df = arp_mod.load_training_data(
        tracker,
        evolution_path=tmp_path / "evo.db",
        roi_events_path=tmp_path / "roi_events.db",
        output_path=tmp_path / "out.csv",
        router=db_router.GLOBAL_ROUTER,
    )
    conn = db_router.GLOBAL_ROUTER.get_connection("roi_prediction_events")
    assert conn is not None
    assert "prediction_confidence" in df.columns
    assert df["prediction_confidence"].iloc[0] == pytest.approx(0.0)


def test_horizon_specific_prediction_logging(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_router.init_db_router(
        "roi2", local_db_path=str(tmp_path / "roi_events.db"), shared_db_path=str(tmp_path / "shared.db")
    )
    importlib.reload(arp_mod)
    importlib.reload(tb_mod)
    importlib.reload(roi_mod)
    tb_mod._init_roi_events_db()
    tracker = roi_mod.ROITracker()
    predicted = [1.0, 2.0, 3.0]
    actual = [0.9, 1.8, 2.7]
    tracker.record_prediction(
        predicted,
        actual,
        predicted_class="linear",
        actual_class="linear",
    )
    conn = db_router.GLOBAL_ROUTER.get_connection("roi_prediction_events")
    assert conn is not None
    tracker.evaluate_model(window=1, roi_events_path=str(tmp_path / "roi_events.db"))
    mae_hist = tracker.horizon_mae_history[-1]
    assert isinstance(mae_hist, dict)


def test_prediction_logging_and_accuracy(tmp_path, monkeypatch):
    """Predictions store sequences and categories and report accuracy."""

    tracker = roi_mod.ROITracker()
    monkeypatch.chdir(tmp_path)

    tracker.record_prediction(
        [1.0, 2.0],
        [1.0, 2.0],
        predicted_class="linear",
        actual_class="linear",
    )
    tracker.record_prediction(
        [2.0, 3.0],
        [2.0, 3.0],
        predicted_class="linear",
        actual_class="exponential",
    )

    conn = sqlite3.connect(tmp_path / "roi_events.db")
    rows = conn.execute(
        "SELECT predicted_horizons, predicted_categories, actual_categories FROM roi_prediction_events ORDER BY ts"
    ).fetchall()
    conn.close()
    assert json.loads(rows[0][0]) == [1.0, 2.0]
    assert json.loads(rows[0][1]) == ["linear"]
    assert json.loads(rows[1][1]) == ["linear"]
    assert json.loads(rows[1][2]) == ["exponential"]

    assert tracker.classification_accuracy() == pytest.approx(0.5)
    acc, mae = tracker.evaluate_model(
        window=2, roi_events_path=str(tmp_path / "roi_events.db")
    )
    assert acc == pytest.approx(0.5)
    assert mae == pytest.approx(0.0)


def test_online_update(monkeypatch):
    """``update`` performs incremental fitting when supported."""

    X = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([[0.0], [1.0]], dtype=float)
    g = np.array(["linear", "linear"], dtype=object)

    def fake_build(*_args, **kwargs):
        if kwargs.get("return_feature_names"):
            return X, y, g, []
        return X, y, g

    class DummyModel:
        def __init__(self, **_kwargs):
            self.partial_calls = 0
            self.dim = 1

        def fit(self, X, y):  # pragma: no cover - called during init
            self.dim = y.shape[1] if y.ndim > 1 else 1
            return self

        def partial_fit(self, X, y):
            self.partial_calls += 1
            return self.fit(X, y)

        def predict(self, X):  # pragma: no cover - simple stub
            return np.zeros((len(X), self.dim))

        def get_params(self):  # pragma: no cover - not used
            return {}

    monkeypatch.setattr(
        "menace_sandbox.adaptive_roi_predictor.build_dataset", fake_build
    )

    def stub_train(self, dataset=None, **_kwargs):  # pragma: no cover - setup
        if dataset is None:
            dataset = fake_build()
        Xd, yd, gd = dataset
        self._model = DummyModel()
        self._model.fit(Xd, yd)
        self.training_data = (Xd, yd, gd)
        self._trained_size = len(Xd)

    monkeypatch.setattr(arp_mod.AdaptiveROIPredictor, "train", stub_train)

    predictor = arp_mod.AdaptiveROIPredictor(cv=0, param_grid={})
    assert isinstance(predictor._model, DummyModel)
    base_size = predictor._trained_size

    def fail_train(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("retrain should not be called")

    monkeypatch.setattr(arp_mod.AdaptiveROIPredictor, "train", fail_train)

    predictor.update([[2.0]], [[2.0]], ["linear"])

    assert predictor._trained_size == base_size + 1
    assert predictor.training_data is not None
    assert predictor.training_data[0].shape[0] == base_size + 1
    assert predictor._model.partial_calls == 1

