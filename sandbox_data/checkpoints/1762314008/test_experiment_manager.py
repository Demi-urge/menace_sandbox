import os
import types
import sys
import asyncio
from pathlib import Path
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

vs = types.ModuleType("vector_service")
class DummyBuilder:
    def __init__(self, *a, **k):
        pass
    def refresh_db_weights(self):
        pass
    def build_context(self, *a, **k):
        return {}
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

import sqlite3
import db_router
db_router.GLOBAL_ROUTER = types.SimpleNamespace(
    menace_id="test",
    get_connection=lambda *a, **k: sqlite3.connect(":memory:")
)
from menace.experiment_manager import ExperimentManager, ExperimentResult
from menace import experiment_history_db as exp_hist_mod
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent

class DummyDataBot:
    def __init__(self):
        self.db = types.SimpleNamespace(fetch=lambda limit=1: types.SimpleNamespace(empty=True))

class DummyCapitalBot:
    pass


def test_creates_pipeline_with_builder(monkeypatch):
    import menace.experiment_manager as exp_mod

    class DummyPipeline:
        def __init__(self, data_bot, capital_manager, context_builder):
            self.context_builder = context_builder

        def run(self, name, energy=1):
            return types.SimpleNamespace(roi=None)

    monkeypatch.setattr(exp_mod, "ModelAutomationPipeline", DummyPipeline)

    builder = DummyBuilder()
    mgr = ExperimentManager(DummyDataBot(), DummyCapitalBot(), context_builder=builder)
    assert isinstance(mgr.pipeline, DummyPipeline)
    assert mgr.pipeline.context_builder is builder


def test_pipeline_builder_mismatch(monkeypatch):
    builder = DummyBuilder()
    other = DummyBuilder()
    bad_pipeline = types.SimpleNamespace(run=lambda *a, **k: None, context_builder=other)

    with pytest.raises(ValueError):
        ExperimentManager(
            DummyDataBot(),
            DummyCapitalBot(),
            pipeline=bad_pipeline,
            context_builder=builder,
        )

def test_best_variant_significant(tmp_path):
    db_router.init_db_router(
        "test", local_db_path=str(tmp_path / "l.db"), shared_db_path=str(tmp_path / "s.db")
    )
    importlib.reload(exp_hist_mod)
    db = exp_hist_mod.ExperimentHistoryDB()
    builder = DummyBuilder()
    dummy_pipeline = types.SimpleNamespace(run=lambda *a, **k: None, context_builder=builder)
    mgr = ExperimentManager(
        DummyDataBot(),
        DummyCapitalBot(),
        pipeline=dummy_pipeline,
        experiment_db=db,
        context_builder=builder,
    )
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
    builder = DummyBuilder()
    dummy_pipeline = types.SimpleNamespace(run=lambda *a, **k: None, context_builder=builder)
    mgr = ExperimentManager(
        DummyDataBot(),
        DummyCapitalBot(),
        pipeline=dummy_pipeline,
        experiment_db=db,
        context_builder=builder,
    )
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
        def __init__(self, builder):
            self.context_builder = builder
        def run(self, name, energy=1):
            return types.SimpleNamespace(roi=types.SimpleNamespace(roi=1.0))

    db_router.init_db_router(
        "test3", local_db_path=str(tmp_path / "l3.db"), shared_db_path=str(tmp_path / "s3.db")
    )
    importlib.reload(exp_hist_mod)
    builder = DummyBuilder()
    mgr = ExperimentManager(
        DummyDataBot(),
        DummyCapitalBot(),
        pipeline=DummyPipeline(builder),
        experiment_db=exp_hist_mod.ExperimentHistoryDB(),
        lineage=types.SimpleNamespace(history_db=hist),
        context_builder=builder,
    )
    res = asyncio.run(mgr.run_experiments_from_parent(root_id))
    assert {r.variant for r in res} == {"A", "B"}
