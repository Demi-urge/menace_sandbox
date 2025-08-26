from types import ModuleType, SimpleNamespace
import importlib.util
import sys


def _dummy() -> bool:
    return True


def test_stability_gate(monkeypatch):
    pkg = ModuleType("menace_sandbox")
    pkg.__path__ = []  # mark as package
    sys.modules["menace_sandbox"] = pkg

    cws_mod = ModuleType("menace_sandbox.composite_workflow_scorer")

    class CompositeWorkflowScorer:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            return SimpleNamespace(roi_gain=0.0, runtime=0.0, success_rate=1.0)
    cws_mod.CompositeWorkflowScorer = CompositeWorkflowScorer
    sys.modules["menace_sandbox.composite_workflow_scorer"] = cws_mod

    bot_mod = ModuleType("menace_sandbox.workflow_evolution_bot")

    class WorkflowEvolutionBot:
        _rearranged_events = {}

        def generate_variants(self, limit, workflow_id):
            return []
    bot_mod.WorkflowEvolutionBot = WorkflowEvolutionBot
    sys.modules["menace_sandbox.workflow_evolution_bot"] = bot_mod

    db_mod = ModuleType("menace_sandbox.roi_results_db")

    class ROIResultsDB:
        def __init__(self, *a, **k):
            pass

        def log_module_delta(self, *a, **k):
            pass
    db_mod.ROIResultsDB = ROIResultsDB
    sys.modules["menace_sandbox.roi_results_db"] = db_mod

    mut_mod = ModuleType("menace_sandbox.mutation_logger")

    def log_mutation(*a, **k):
        pass

    def record_mutation_outcome(*a, **k):
        pass
    mut_mod.log_mutation = log_mutation
    mut_mod.record_mutation_outcome = record_mutation_outcome
    sys.modules["menace_sandbox.mutation_logger"] = mut_mod

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.workflow_evolution_manager",
        "workflow_evolution_manager.py",
    )
    wem = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.workflow_evolution_manager"] = wem
    assert spec.loader is not None
    spec.loader.exec_module(wem)

    wem._roi_delta_ema.clear()
    wem._gating_counts.clear()
    wem.GATING_THRESHOLD = 0.1

    monkeypatch.setattr(
        wem.WorkflowEvolutionBot,
        "generate_variants",
        lambda self, limit, workflow_id: ["x"],
    )

    side_effects = [1.0, 0.5] * wem.GATING_CONSECUTIVE

    def fake_run(self, workflow_callable, wf_id_str, run_id):
        roi_gain = side_effects.pop(0)
        return SimpleNamespace(roi_gain=roi_gain, runtime=0.0, success_rate=1.0)

    monkeypatch.setattr(wem.CompositeWorkflowScorer, "run", fake_run)

    wf_id = 1
    for _ in range(wem.GATING_CONSECUTIVE):
        wem.evolve(_dummy, wf_id, variants=1)

    assert wem.is_stable(wf_id)
