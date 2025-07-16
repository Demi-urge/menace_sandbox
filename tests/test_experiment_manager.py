import types
import sys

dummy_mods = {
    'menace.model_automation_pipeline': types.SimpleNamespace(ModelAutomationPipeline=object, AutomationResult=object),
    'menace.data_bot': types.SimpleNamespace(DataBot=object),
    'menace.capital_management_bot': types.SimpleNamespace(CapitalManagementBot=object),
    'menace.prediction_manager_bot': types.SimpleNamespace(PredictionManager=object),
}
sys.modules.update(dummy_mods)

from menace.experiment_manager import ExperimentManager, ExperimentResult
from menace.experiment_history_db import ExperimentHistoryDB

class DummyDataBot:
    def __init__(self):
        self.db = types.SimpleNamespace(fetch=lambda limit=1: types.SimpleNamespace(empty=True))

class DummyCapitalBot:
    pass

def test_best_variant_significant(tmp_path):
    db = ExperimentHistoryDB(tmp_path / "e.db")
    dummy_pipeline = types.SimpleNamespace(run=lambda *a, **k: None)
    mgr = ExperimentManager(DummyDataBot(), DummyCapitalBot(), pipeline=dummy_pipeline, experiment_db=db)
    res = [
        ExperimentResult("A", 1.0, {}, sample_size=30, variance=1.0),
        ExperimentResult("B", 0.0, {}, sample_size=30, variance=1.0),
    ]
    best = mgr.best_variant(res)
    assert best and best.variant == "A"

def test_best_variant_not_significant(tmp_path):
    db = ExperimentHistoryDB(tmp_path / "e2.db")
    dummy_pipeline = types.SimpleNamespace(run=lambda *a, **k: None)
    mgr = ExperimentManager(DummyDataBot(), DummyCapitalBot(), pipeline=dummy_pipeline, experiment_db=db)
    res = [
        ExperimentResult("A", 1.0, {}, sample_size=30, variance=1.0),
        ExperimentResult("B", 0.99, {}, sample_size=30, variance=1.0),
    ]
    best = mgr.best_variant(res)
    assert best is None
