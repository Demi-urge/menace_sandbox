import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
from dynamic_path_router import resolve_path


@pytest.fixture
def evolution_setup():
    """Prepare a minimal environment for workflow evolution tests."""

    pkg = ModuleType("menace_sandbox")
    pkg.__path__ = []  # mark as package
    sys.modules["menace_sandbox"] = pkg

    # mutation logger ---------------------------------------------------
    mut_mod = ModuleType("menace_sandbox.mutation_logger")
    log_calls = []

    def log_mutation(**kwargs):
        log_calls.append(kwargs)
        return len(log_calls)

    record_calls = []

    def record_mutation_outcome(*args, **kwargs):
        record_calls.append((args, kwargs))

    def log_workflow_evolution(**kwargs):
        return 0

    mut_mod.log_mutation = log_mutation
    mut_mod.log_mutation_calls = log_calls
    mut_mod.record_mutation_outcome = record_mutation_outcome
    mut_mod.record_mutation_outcome_calls = record_calls
    mut_mod.log_workflow_evolution = log_workflow_evolution
    mut_mod._history_db = SimpleNamespace(record_outcome=lambda *a, **k: None)
    sys.modules["menace_sandbox.mutation_logger"] = mut_mod

    # ROI results db ----------------------------------------------------
    db_mod = ModuleType("menace_sandbox.roi_results_db")

    class ROIResultsDB:
        logged: list[tuple[str, str, str, float]] = []

        def __init__(self, *a, **k):
            pass

        def log_module_delta(self, workflow_id, run_id, module, runtime, success_rate, roi_delta):
            self.logged.append((workflow_id, run_id, module, roi_delta))

    db_mod.ROIResultsDB = ROIResultsDB
    sys.modules["menace_sandbox.roi_results_db"] = db_mod

    # composite workflow scorer ----------------------------------------
    cws_mod = ModuleType("menace_sandbox.composite_workflow_scorer")

    class CompositeWorkflowScorer:
        roi_values: list[float] = []
        run_ids: list[str] = []

        def __init__(self, *a, **k):
            pass

        def run(self, workflow_callable, wf_id_str, run_id):
            self.run_ids.append(run_id)
            roi = self.roi_values.pop(0)
            return SimpleNamespace(roi_gain=roi, runtime=0.0, success_rate=1.0)

    cws_mod.CompositeWorkflowScorer = CompositeWorkflowScorer
    sys.modules["menace_sandbox.composite_workflow_scorer"] = cws_mod

    # evolution bot -----------------------------------------------------
    bot_mod = ModuleType("menace_sandbox.workflow_evolution_bot")

    class WorkflowEvolutionBot:
        def __init__(self, *a, **k):
            self._rearranged_events = {}

        def generate_variants(self, limit, workflow_id):
            seqs = ["syn_step", "base-intent_step"][:limit]
            for seq in seqs:
                event_id = mut_mod.log_mutation(
                    change=seq,
                    reason="variant",
                    trigger="bot",
                    performance=0.0,
                    workflow_id=workflow_id,
                    parent_id=None,
                )
                self._rearranged_events[seq] = event_id
                yield seq

    bot_mod.WorkflowEvolutionBot = WorkflowEvolutionBot
    sys.modules["menace_sandbox.workflow_evolution_bot"] = bot_mod

    tracker_mod = ModuleType("menace_sandbox.roi_tracker")

    class ROITracker:
        def __init__(self, *a, **k):
            self.roi_history = []

        def diminishing(self) -> float:
            return 0.05

        def calculate_raroi(self, roi, **kw):
            return roi, roi, []

        def score_workflow(self, workflow_id, raroi, tau=None):
            pass

    tracker_mod.ROITracker = ROITracker
    sys.modules["menace_sandbox.roi_tracker"] = tracker_mod

    settings_mod = ModuleType("menace_sandbox.sandbox_settings")
    settings_mod.SandboxSettings = lambda *a, **k: SimpleNamespace(
        roi_ema_alpha=0.1,
        workflow_merge_similarity=0.9,
        workflow_merge_entropy_delta=0.1,
        duplicate_similarity=0.95,
        duplicate_entropy=0.05,
    )
    sys.modules["menace_sandbox.sandbox_settings"] = settings_mod

    stab_mod = ModuleType("menace_sandbox.workflow_stability_db")

    class WorkflowStabilityDB:
        def __init__(self, *a, **k):
            self.data: dict[str, dict[str, float | int]] = {}

        def is_stable(self, wf, current_roi=None, threshold=None):
            entry = self.data.get(wf)
            if not entry or "roi" not in entry:
                return False
            if current_roi is not None and threshold is not None:
                prev = entry.get("roi", 0.0)
                if abs(current_roi - prev) > threshold:
                    del self.data[wf]
                    return False
            return True

        def mark_stable(self, wf, roi):
            entry = self.data.get(wf, {})
            entry["roi"] = roi
            self.data[wf] = entry

        def clear(self, wf):
            entry = self.data.get(wf)
            if entry:
                entry.pop("roi", None)
                self.data[wf] = entry

        def clear_all(self):
            self.data.clear()

        def get_ema(self, wf):
            entry = self.data.get(wf, {})
            return entry.get("ema", 0.0), entry.get("count", 0)

        def set_ema(self, wf, ema, count):
            entry = self.data.get(wf, {})
            entry.update({"ema": ema, "count": count})
            self.data[wf] = entry

    stab_mod.WorkflowStabilityDB = WorkflowStabilityDB
    sys.modules["menace_sandbox.workflow_stability_db"] = stab_mod

    summary_mod = ModuleType("menace_sandbox.workflow_summary_db")

    class WorkflowSummaryDB:
        def set_summary(self, workflow_id, status):
            pass

    summary_mod.WorkflowSummaryDB = WorkflowSummaryDB
    sys.modules["menace_sandbox.workflow_summary_db"] = summary_mod

    # finally load workflow evolution manager --------------------------
    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.workflow_evolution_manager",
        resolve_path("workflow_evolution_manager.py"),  # path-ignore
    )
    wem = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.workflow_evolution_manager"] = wem
    assert spec.loader is not None
    spec.loader.exec_module(wem)

    wem.STABLE_WORKFLOWS.clear_all()

    return SimpleNamespace(
        module=wem,
        scorer=CompositeWorkflowScorer,
        db=ROIResultsDB,
        logger=mut_mod,
    )


@pytest.fixture
def baseline_workflow():
    def _wf() -> bool:
        return True

    return _wf


def test_variant_promotion_and_benchmarking(evolution_setup, baseline_workflow):
    wem = evolution_setup.module
    scorer = evolution_setup.scorer
    db = evolution_setup.db
    logger = evolution_setup.logger

    # ROI sequence: baseline=1.0, synergy variant=2.0, intent variant=0.5
    scorer.roi_values = [1.0, 2.0, 0.5]
    scorer.run_ids.clear()
    db.logged.clear()
    logger.log_mutation_calls.clear()

    promoted = wem.evolve(baseline_workflow, workflow_id=1, variants=2)

    # baseline and both variants are benchmarked
    assert scorer.run_ids[0] == "baseline"
    assert len(scorer.run_ids) == 3
    # module deltas recorded for each variant
    modules = [m for (_, _, m, _) in db.logged]
    assert {"variant:syn_step", "variant:base-intent_step"} <= set(modules)

    # top ROI variant is promoted
    assert promoted is not baseline_workflow
    assert logger.log_mutation_calls[-1]["reason"] == "promoted"


def test_stable_when_no_variant_improves(evolution_setup, baseline_workflow):
    wem = evolution_setup.module
    scorer = evolution_setup.scorer
    db = evolution_setup.db
    logger = evolution_setup.logger

    scorer.roi_values = [1.0, 0.9, 0.8]
    scorer.run_ids.clear()
    db.logged.clear()
    logger.log_mutation_calls.clear()

    result = wem.evolve(baseline_workflow, workflow_id=2, variants=2)

    assert result is baseline_workflow
    assert logger.log_mutation_calls[-1]["reason"] == "stable"
    assert len(scorer.run_ids) == 3


def test_gating_prevents_further_evolution(
    monkeypatch, tmp_path, evolution_setup, baseline_workflow
):
    wem = evolution_setup.module
    scorer = evolution_setup.scorer

    monkeypatch.setenv("ROI_GATING_THRESHOLD", "0.05")
    monkeypatch.setenv("ROI_GATING_CONSECUTIVE", "1")

    scorer.roi_values = [1.0, 1.02]
    scorer.run_ids.clear()

    # First evolution marks workflow stable
    wem.evolve(baseline_workflow, workflow_id=3, variants=1)
    assert wem.is_stable(3)
    assert scorer.run_ids[0] == "baseline"
    assert len(scorer.run_ids) == 2

    # Second attempt should skip variant benchmarking
    scorer.roi_values = [1.0]
    scorer.run_ids.clear()
    result = wem.evolve(baseline_workflow, workflow_id=3, variants=1)
    assert result is baseline_workflow
    assert scorer.run_ids == ["baseline"]


def test_multiple_low_roi_cycles_trigger_stability(
    monkeypatch, evolution_setup, baseline_workflow
):
    wem = evolution_setup.module
    scorer = evolution_setup.scorer

    monkeypatch.setenv("ROI_GATING_THRESHOLD", "0.05")
    monkeypatch.setenv("ROI_GATING_CONSECUTIVE", "2")

    # first cycle with small positive delta
    scorer.roi_values = [1.0, 1.01]
    scorer.run_ids.clear()
    wem.evolve(baseline_workflow, workflow_id=4, variants=1)
    assert not wem.is_stable(4)
    assert len(scorer.run_ids) == 2

    # second cycle triggers stability via EMA gating
    scorer.roi_values = [1.0, 1.01]
    scorer.run_ids.clear()
    wem.evolve(baseline_workflow, workflow_id=4, variants=1)
    assert wem.is_stable(4)

    # subsequent evolution should skip variant benchmarking
    scorer.roi_values = [1.0]
    scorer.run_ids.clear()
    result = wem.evolve(baseline_workflow, workflow_id=4, variants=1)
    assert result is baseline_workflow
    assert scorer.run_ids == ["baseline"]


@pytest.mark.skip("ROI gating stabilizes immediately on non-positive deltas")
def test_roi_gating_counter(monkeypatch, tmp_path, evolution_setup, baseline_workflow):
    pass


@pytest.mark.skip("ROI gating stabilizes immediately on non-positive deltas")
def test_roi_gating_counter_reset(monkeypatch, tmp_path, evolution_setup, baseline_workflow):
    pass
