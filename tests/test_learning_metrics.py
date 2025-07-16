import pytest

pytest.importorskip("prometheus_client")

from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.menace_memory_manager import MenaceMemoryManager
from menace.learning_engine import LearningEngine
from menace.metrics_exporter import (
    learning_cv_score,
    learning_holdout_score,
    security_score_gauge,
    safety_rating_gauge,
    adaptability_gauge,
    antifragility_gauge,
    shannon_entropy_gauge,
    efficiency_gauge,
    flexibility_gauge,
    projected_lucrativity_gauge,
)


def test_learning_metrics_export(tmp_path):
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
    engine = LearningEngine(pdb, mm, persist_path=tmp_path / "scores.json")
    engine.train()
    result = engine.evaluate()
    engine.persist_evaluation(result)
    assert learning_cv_score is not None
    assert learning_holdout_score is not None
    assert learning_cv_score._value.get() == pytest.approx(result["cv_score"])
    assert learning_holdout_score._value.get() == pytest.approx(result["holdout_score"])


def test_extended_gauges_available():
    assert security_score_gauge is not None
    assert safety_rating_gauge is not None
    assert adaptability_gauge is not None
    assert antifragility_gauge is not None
    assert shannon_entropy_gauge is not None
    assert efficiency_gauge is not None
    assert flexibility_gauge is not None
    assert projected_lucrativity_gauge is not None

