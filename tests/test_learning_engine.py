from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.menace_memory_manager import MenaceMemoryManager
from menace.learning_engine import LearningEngine, load_score_history
import json
import pytest
import threading
import time



def test_train_and_predict(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    engine = LearningEngine(pdb, mm)
    trained = engine.train()
    assert trained
    prob = engine.predict_success(1.0, 1.0, 1.0, 1.0, "A")
    assert 0.0 <= prob <= 1.0


def test_train_and_predict_nn(tmp_path):
    pytest.importorskip("torch")
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    engine = LearningEngine(pdb, mm, model="nn")
    trained = engine.train()
    assert trained
    prob = engine.predict_success(1.0, 1.0, 1.0, 1.0, "A")
    assert 0.0 <= prob <= 1.0


def test_train_and_predict_lstm(tmp_path):
    pytest.importorskip("torch")
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    engine = LearningEngine(pdb, mm, model="lstm")
    trained = engine.train()
    assert trained
    prob = engine.predict_success(1.0, 1.0, 1.0, 1.0, "A")
    assert 0.0 <= prob <= 1.0


def test_train_and_predict_transfer(tmp_path):
    pytest.importorskip("transformers")
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    engine = LearningEngine(pdb, mm, model="bert-transfer")
    try:
        trained = engine.train()
    except RuntimeError:
        pytest.skip("transformers not available")
    assert trained
    prob = engine.predict_success(1.0, 1.0, 1.0, 1.0, "A")
    assert 0.0 <= prob <= 1.0


def test_train_and_predict_transformer(tmp_path):
    pytest.importorskip("transformers")
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    engine = LearningEngine(pdb, mm, model="transformer")
    try:
        trained = engine.train()
    except RuntimeError:
        pytest.skip("transformers not available")
    assert trained
    prob = engine.predict_success(1.0, 1.0, 1.0, 1.0, "A")
    assert 0.0 <= prob <= 1.0


def test_evaluation_persistence_json(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    path = tmp_path / "scores.json"
    engine = LearningEngine(pdb, mm, persist_path=path)
    engine.train()
    res = engine.evaluate()
    engine.persist_evaluation(res)
    hist = load_score_history(path)
    assert hist
    assert hist[-1]["cv_score"] == res["cv_score"]


def test_evaluation_persistence_sqlite(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    dbpath = tmp_path / "scores.db"
    engine = LearningEngine(pdb, mm, persist_path=dbpath)
    engine.train()
    res = engine.evaluate()
    engine.persist_evaluation(res)
    hist = load_score_history(dbpath)
    assert hist
    assert hist[-1]["holdout_score"] == res["holdout_score"]


def test_auto_train_persists_best_model(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    path = tmp_path / "scores.json"
    engine = LearningEngine(pdb, mm, persist_path=path)
    best = engine.auto_train(models=["logreg"])
    assert best
    cfg = json.load(open(tmp_path / "best_model.json", "r", encoding="utf-8"))
    assert cfg["model"] == "logreg"
    engine2 = LearningEngine(pdb, mm, persist_path=path)
    best2 = engine2.auto_train(models=["logreg"])
    assert best2 == "logreg"


def test_auto_train_repeated_eval(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    pdb.log(
        PathwayRecord(
            actions="A",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )
    path = tmp_path / "scores.json"
    engine = LearningEngine(pdb, mm, persist_path=path)

    th = threading.Thread(
        target=engine.auto_train,
        kwargs={"models": ["logreg"], "eval_interval": 0.01},
        daemon=True,
    )
    th.start()
    time.sleep(0.05)
    engine.stop_auto_train()
    th.join(timeout=1)

    hist = load_score_history(path)
    assert len(hist) >= 2
