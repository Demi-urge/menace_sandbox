import sys
import types

# Minimal stub for UnifiedEventBus to avoid heavy dependencies during import
bus_stub = types.ModuleType("menace.unified_event_bus")
class DummyBus:
    def __init__(self, *a, **k):
        pass
    def publish(self, *a, **k):
        pass
    def subscribe(self, *a, **k):
        pass
    def subscribe_async(self, *a, **k):
        pass
bus_stub.UnifiedEventBus = DummyBus
sys.modules.setdefault("menace.unified_event_bus", bus_stub)

from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.workflow_evolution_bot import WorkflowEvolutionBot
from menace import mutation_logger as ml
from menace.evolution_history_db import EvolutionHistoryDB


def test_rearranged_workflow_lineage(tmp_path, monkeypatch):
    pdb = PathwayDB(tmp_path / "p.db")
    pdb.log(
        PathwayRecord(
            actions="1-2-3",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.record_sequence([1, 2, 3])
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    monkeypatch.setattr(ml, "_history_db", hist)
    monkeypatch.setattr(ml, "_event_bus", None)
    root_id = ml.log_mutation("1-2-3", "original", "test", 0.0, workflow_id=1)
    dummy_clusterer = types.SimpleNamespace(find_modules_related_to=lambda *a, **k: [])
    bot = WorkflowEvolutionBot(pdb, intent_clusterer=dummy_clusterer)
    seqs = list(
        bot.generate_variants(limit=1, workflow_id=1, parent_event_id=root_id)
    )
    assert seqs
    variant = seqs[0]
    bot.record_benchmark(variant, after_metric=1.0, roi=0.5, performance=0.1)
    tree = ml.build_lineage(1)
    assert tree and tree[0]["rowid"] == root_id
    assert tree[0]["children"] and tree[0]["children"][0]["action"] == variant
