from types import ModuleType, SimpleNamespace
import importlib.util
import sys


def _load_manager(variant_seq="mod_a", variant_roi=1.5):
    # Prepare package namespace so relative imports inside the module resolve
    pkg = ModuleType("menace_sandbox")
    pkg.__path__ = []  # mark as package
    sys.modules["menace_sandbox"] = pkg

    # Stub CompositeWorkflowScorer used for benchmarking
    cws_mod = ModuleType("menace_sandbox.composite_workflow_scorer")

    class CompositeWorkflowScorer:
        def __init__(self, *a, **k):
            pass

        def run(self, workflow_callable, wf_id_str, run_id):
            workflow_callable()
            roi_gain = 1.0 if run_id == "baseline" else variant_roi
            return SimpleNamespace(roi_gain=roi_gain, runtime=0.0, success_rate=1.0)

    cws_mod.CompositeWorkflowScorer = CompositeWorkflowScorer
    sys.modules["menace_sandbox.composite_workflow_scorer"] = cws_mod

    # Stub WorkflowEvolutionBot to produce a single variant sequence
    bot_mod = ModuleType("menace_sandbox.workflow_evolution_bot")

    class WorkflowEvolutionBot:
        _rearranged_events = {}

        def generate_variants(self, limit, workflow_id):
            return [variant_seq]

    bot_mod.WorkflowEvolutionBot = WorkflowEvolutionBot
    sys.modules["menace_sandbox.workflow_evolution_bot"] = bot_mod

    # Stub ROIResultsDB to capture benchmarking deltas
    db_calls = []

    class ROIResultsDB:
        def __init__(self, *a, **k):
            pass

        def log_module_delta(self, *a, **k):
            db_calls.append((a, k))

    db_mod = ModuleType("menace_sandbox.roi_results_db")
    db_mod.ROIResultsDB = ROIResultsDB
    sys.modules["menace_sandbox.roi_results_db"] = db_mod

    # Stub mutation logger to observe promotion events
    mut_calls = []

    def log_mutation(*a, **k):
        mut_calls.append(("log", k))

    def record_mutation_outcome(*a, **k):
        mut_calls.append(("record", k))

    mut_mod = ModuleType("menace_sandbox.mutation_logger")
    mut_mod.log_mutation = log_mutation
    mut_mod.record_mutation_outcome = record_mutation_outcome
    sys.modules["menace_sandbox.mutation_logger"] = mut_mod

    # Load the workflow_evolution_manager module under the package namespace
    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.workflow_evolution_manager", "workflow_evolution_manager.py"
    )
    wem = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.workflow_evolution_manager"] = wem
    assert spec.loader is not None
    spec.loader.exec_module(wem)
    wem._roi_delta_ema.clear()
    wem._gating_counts.clear()
    return wem, db_calls, mut_calls


def test_variant_generation_and_benchmarking():
    run_log = []
    mod = ModuleType("mod_a")
    mod.run = lambda: run_log.append("variant") or True  # type: ignore
    sys.modules["mod_a"] = mod

    wem, db_calls, _ = _load_manager(variant_seq="mod_a", variant_roi=1.5)

    baseline = lambda: run_log.append("baseline") or True
    wem.evolve(baseline, workflow_id=1, variants=1)

    assert "baseline" in run_log
    assert "variant" in run_log  # variant callable executed
    assert db_calls, "benchmarking results were not logged"
    _, kwargs = db_calls[0]
    assert kwargs["module"] == "variant:mod_a"
    assert kwargs["roi_delta"] == 0.5


def test_promotion_logic():
    run_log = []
    mod = ModuleType("mod_b")
    mod.run = lambda: run_log.append("variant") or True  # type: ignore
    sys.modules["mod_b"] = mod

    wem, _, mut_calls = _load_manager(variant_seq="mod_b", variant_roi=2.0)

    baseline = lambda: run_log.append("baseline") or True
    promoted = wem.evolve(baseline, workflow_id=1, variants=1)

    assert any(c[0] == "log" and c[1]["reason"] == "promoted" for c in mut_calls)

    run_log.clear()
    promoted()
    assert run_log == ["variant"]
