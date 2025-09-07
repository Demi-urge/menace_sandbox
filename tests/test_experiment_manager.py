import os
import types
import sys
import asyncio
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

vs = types.ModuleType("vector_service")
class DummyBuilder:
    def __init__(self, *a, **k):
        pass
    def refresh_db_weights(self):
        pass
vs.ContextBuilder = DummyBuilder
vs.CognitionLayer = object
sys.modules["vector_service"] = vs

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(Path(__file__).resolve().parent.parent)]
menace_pkg.RAISE_ERRORS = False
sys.modules["menace"] = menace_pkg
sys.path.append(str(Path(__file__).resolve().parent.parent))

dummy_mods = {
    'menace.model_automation_pipeline': types.SimpleNamespace(ModelAutomationPipeline=object, AutomationResult=object),
    'menace.data_bot': types.SimpleNamespace(DataBot=object),
    'menace.capital_management_bot': types.SimpleNamespace(CapitalManagementBot=object),
    'menace.prediction_manager_bot': types.SimpleNamespace(PredictionManager=object),
    'menace.mutation_lineage': types.SimpleNamespace(MutationLineage=object),
    'menace.patch_provenance': types.SimpleNamespace(get_patch_provenance=lambda *a, **k: None),
}
sys.modules.update(dummy_mods)

import importlib

import db_router
from menace.experiment_manager import ExperimentManager, ExperimentResult
from menace import experiment_history_db as exp_hist_mod
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent

class DummyDataBot:
    def __init__(self):
        self.db = types.SimpleNamespace(fetch=lambda limit=1: types.SimpleNamespace(empty=True))

class DummyCapitalBot:
    pass

def test_best_variant_significant(tmp_path):
    db_router.init_db_router(
        "test", local_db_path=str(tmp_path / "l.db"), shared_db_path=str(tmp_path / "s.db")
    )
    importlib.reload(exp_hist_mod)
    db = exp_hist_mod.ExperimentHistoryDB()
    dummy_pipeline = types.SimpleNamespace(run=lambda *a, **k: None)
    mgr = ExperimentManager(DummyDataBot(), DummyCapitalBot(), pipeline=dummy_pipeline, experiment_db=db)
    res = [
        ExperimentResult("A", 1.0, {}, sample_size=30, variance=1.0),
        ExperimentResult("B", 0.0, {}, sample_size=30, variance=1.0),
    ]
    best = mgr.best_variant(res)
    assert best and best.variant == "A"

def test_best_variant_not_significant(tmp_path):
    db_router.init_db_router(
        "test2", local_db_path=str(tmp_path / "l2.db"), shared_db_path=str(tmp_path / "s2.db")
    )
    importlib.reload(exp_hist_mod)
    db = exp_hist_mod.ExperimentHistoryDB()
    dummy_pipeline = types.SimpleNamespace(run=lambda *a, **k: None)
    mgr = ExperimentManager(DummyDataBot(), DummyCapitalBot(), pipeline=dummy_pipeline, experiment_db=db)
    res = [
        ExperimentResult("A", 1.0, {}, sample_size=30, variance=1.0),
        ExperimentResult("B", 0.99, {}, sample_size=30, variance=1.0),
    ]
    best = mgr.best_variant(res)
    assert best is None


def test_run_experiments_from_parent(tmp_path):
    hist = EvolutionHistoryDB(tmp_path / "h.db")
    root_id = hist.add(EvolutionEvent("root", 0, 1, 1.0, workflow_id=1))
    hist.spawn_variant(root_id, "A")
    hist.spawn_variant(root_id, "B")

    class DummyPipeline:
        def run(self, name, energy=1):
            return types.SimpleNamespace(roi=types.SimpleNamespace(roi=1.0))

    db_router.init_db_router(
        "test3", local_db_path=str(tmp_path / "l3.db"), shared_db_path=str(tmp_path / "s3.db")
    )
    importlib.reload(exp_hist_mod)
    mgr = ExperimentManager(
        DummyDataBot(),
        DummyCapitalBot(),
        pipeline=DummyPipeline(),
        experiment_db=exp_hist_mod.ExperimentHistoryDB(),
        lineage=types.SimpleNamespace(history_db=hist),
    )
    res = asyncio.run(mgr.run_experiments_from_parent(root_id))
    assert {r.variant for r in res} == {"A", "B"}
