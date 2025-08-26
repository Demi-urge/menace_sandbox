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
            return ["x"]

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

    tracker_mod = ModuleType("menace_sandbox.roi_tracker")

    class ROITracker:
        def __init__(self, *a, **k):
            self.roi_history = []

        def diminishing(self) -> float:
            return 0.1

        def calculate_raroi(self, roi, **kw):
            return roi, roi, []

        def score_workflow(self, workflow_id, raroi, tau=None):
            pass

    tracker_mod.ROITracker = ROITracker
    sys.modules["menace_sandbox.roi_tracker"] = tracker_mod

    stab_mod = ModuleType("menace_sandbox.workflow_stability_db")

    class WorkflowStabilityDB:
        def __init__(self, *a, **k):
            self.data: dict[str, float] = {}

        def is_stable(self, wf, current_roi=None, threshold=None):
            if wf not in self.data:
                return False
            if current_roi is not None and threshold is not None:
                prev = self.data[wf]
                if abs(current_roi - prev) > threshold:
                    del self.data[wf]
                    return False
            return True

        def mark_stable(self, wf, roi):
            self.data[wf] = roi

        def clear(self, wf):
            self.data.pop(wf, None)

        def clear_all(self):
            self.data.clear()

    stab_mod.WorkflowStabilityDB = WorkflowStabilityDB
    sys.modules["menace_sandbox.workflow_stability_db"] = stab_mod

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.workflow_evolution_manager",
        "workflow_evolution_manager.py",
    )
    wem = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.workflow_evolution_manager"] = wem
    assert spec.loader is not None
    spec.loader.exec_module(wem)

    wem.STABLE_WORKFLOWS.clear_all()

    side_effects = [1.0, 0.5, 1.0]

    def fake_run(self, workflow_callable, wf_id_str, run_id):
        roi_gain = side_effects.pop(0)
        return SimpleNamespace(roi_gain=roi_gain, runtime=0.0, success_rate=1.0)

    monkeypatch.setattr(wem.CompositeWorkflowScorer, "run", fake_run)

    wf_id = 1
    wem.evolve(_dummy, wf_id, variants=1)

    assert wem.is_stable(wf_id)

    wem.evolve(_dummy, wf_id, variants=1)

    assert side_effects == []
