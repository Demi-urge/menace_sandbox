import numpy as np

from menace.adaptive_roi_dataset import load_adaptive_roi_dataset
from menace.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
from menace.evolution_history_db import EvolutionEvent, EvolutionHistoryDB


def test_dataset_aggregation(tmp_path):
    evo_db_path = tmp_path / "evo.db"
    eval_db_path = tmp_path / "eval.db"
    evo = EvolutionHistoryDB(evo_db_path)
    eva = EvaluationHistoryDB(eval_db_path)

    # create two evolution events for one engine
    evo.add(EvolutionEvent(action="engine", before_metric=0.2, after_metric=0.5, roi=0.3))
    evo.add(EvolutionEvent(action="engine", before_metric=0.5, after_metric=0.7, roi=0.4))

    # one evaluation record tied to the engine
    eva.add(EvaluationRecord(engine="engine", cv_score=0.8, passed=True))

    X, y, passed = load_adaptive_roi_dataset(evo_db_path, eval_db_path)

    assert X.shape == (1, 2)
    assert y.shape == (1,)
    assert passed.tolist() == [1]
    # features should be normalised (mean approximately 0)
    assert np.allclose(X.mean(axis=0), 0.0)
    assert np.allclose(y.mean(), 0.0)
