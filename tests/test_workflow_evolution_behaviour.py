import sys
from types import ModuleType, SimpleNamespace
import importlib.util
from pathlib import Path

import pytest

from workflow_synthesizer import generate_variants, ModuleIOAnalyzer, WorkflowSynthesizer
from dynamic_path_router import resolve_path


# ---------------------------------------------------------------------------
# Helpers

def _load_manager(variant_rois, generate_calls=None, diminishing=0.05):
    """Load workflow_evolution_manager with stubbed dependencies."""
    pkg = ModuleType("menace_sandbox")
    pkg.__path__ = []
    sys.modules["menace_sandbox"] = pkg

    run_map = {
        f"variant-{hash(seq) & 0xFFFFFFFF:x}": roi for seq, roi in variant_rois.items()
    }

    class CompositeWorkflowScorer:
        def __init__(self, *a, **k):
            pass

        def run(self, fn, wf_id, run_id):
            fn()
            roi = 1.0 if run_id == "baseline" else run_map[run_id]
            return SimpleNamespace(roi_gain=roi, runtime=0.0, success_rate=1.0)

    cws_mod = ModuleType("menace_sandbox.composite_workflow_scorer")
    cws_mod.CompositeWorkflowScorer = CompositeWorkflowScorer
    sys.modules["menace_sandbox.composite_workflow_scorer"] = cws_mod

    class WorkflowEvolutionBot:
        _rearranged_events = {}

        def generate_variants(self, limit, workflow_id):
            if generate_calls is not None:
                generate_calls.append(workflow_id)
            return list(variant_rois.keys())

    bot_mod = ModuleType("menace_sandbox.workflow_evolution_bot")
    bot_mod.WorkflowEvolutionBot = WorkflowEvolutionBot
    sys.modules["menace_sandbox.workflow_evolution_bot"] = bot_mod

    class ROIResultsDB:
        def __init__(self, *a, **k):
            pass

        def log_module_delta(self, *a, **k):
            pass

    db_mod = ModuleType("menace_sandbox.roi_results_db")
    db_mod.ROIResultsDB = ROIResultsDB
    sys.modules["menace_sandbox.roi_results_db"] = db_mod

    mut_mod = ModuleType("menace_sandbox.mutation_logger")
    mut_mod.log_mutation = lambda *a, **k: 1
    mut_mod.record_mutation_outcome = lambda *a, **k: None
    sys.modules["menace_sandbox.mutation_logger"] = mut_mod

    class ROITracker:
        def __init__(self, *a, **k):
            pass

        def diminishing(self) -> float:
            return diminishing

        def calculate_raroi(self, roi, **kw):
            return roi, roi, []

        def score_workflow(self, wf_id, raroi, tau=None):
            pass

    tracker_mod = ModuleType("menace_sandbox.roi_tracker")
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

    class WorkflowStabilityDB:
        def __init__(self, *a, **k):
            self.data = {}

        def is_stable(self, wf, current_roi=None, threshold=None):
            entry = self.data.get(wf)
            if not entry:
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
            self.data.pop(wf, None)

        def clear_all(self):
            self.data.clear()

        def get_ema(self, wf):
            entry = self.data.get(wf, {})
            return entry.get("ema", 0.0), entry.get("count", 0)

        def set_ema(self, wf, ema, count):
            entry = self.data.get(wf, {})
            entry.update({"ema": ema, "count": count})
            self.data[wf] = entry

    stab_mod = ModuleType("menace_sandbox.workflow_stability_db")
    stab_mod.WorkflowStabilityDB = WorkflowStabilityDB
    sys.modules["menace_sandbox.workflow_stability_db"] = stab_mod

    class WorkflowSummaryDB:
        def set_summary(self, *a, **k):
            pass

    summary_mod = ModuleType("menace_sandbox.workflow_summary_db")
    summary_mod.WorkflowSummaryDB = WorkflowSummaryDB
    sys.modules["menace_sandbox.workflow_summary_db"] = summary_mod

    graph_mod = ModuleType("menace_sandbox.workflow_graph")
    class WorkflowGraph:
        def add_workflow(self, *a, **k):
            pass

        def add_dependency(self, *a, **k):
            pass

        def update_workflow(self, *a, **k):
            pass
    graph_mod.WorkflowGraph = WorkflowGraph
    sys.modules["menace_sandbox.workflow_graph"] = graph_mod

    evol_hist_mod = ModuleType("menace_sandbox.evolution_history_db")
    class EvolutionHistoryDB:
        def add(self, *a, **k):
            pass
    class EvolutionEvent:
        def __init__(self, *a, **k):
            pass
    evol_hist_mod.EvolutionHistoryDB = EvolutionHistoryDB
    evol_hist_mod.EvolutionEvent = EvolutionEvent
    sys.modules["menace_sandbox.evolution_history_db"] = evol_hist_mod

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.workflow_evolution_manager",
        resolve_path("workflow_evolution_manager.py"),  # path-ignore
    )
    wem = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.workflow_evolution_manager"] = wem
    assert spec.loader is not None
    spec.loader.exec_module(wem)
    wem.STABLE_WORKFLOWS.clear_all()
    return wem


# ---------------------------------------------------------------------------
# Tests

def test_variant_generation_obeys_dependencies(tmp_path, monkeypatch):
    base_src = resolve_path("tests/fixtures/workflow_modules")
    for name in ["mod_a.py", "mod_b.py", "mod_c.py"]:  # path-ignore
        # write modules without extension so ModuleIOAnalyzer resolves them
        (tmp_path / name[:-3]).write_text((base_src / name).read_text())
    monkeypatch.chdir(tmp_path)

    base = ["mod_a", "mod_b", "mod_c"]

    class Swapper:
        def get_synergy_cluster(self, module_name, threshold=0.7, bfs=False):
            if module_name == "mod_a":
                return {"mod_b"}
            if module_name == "mod_b":
                return {"mod_a"}
            return {module_name}

    variants = generate_variants(base, 5, Swapper(), None)
    assert variants
    analyzer = ModuleIOAnalyzer()
    checker = WorkflowSynthesizer()
    for var in variants:
        assert var.index("mod_a") < var.index("mod_b")
        modules = [analyzer.analyze(Path(m)) for m in var]
        steps = checker.resolve_dependencies(modules)
        assert not any(s.unresolved for s in steps)
        assert [s.module for s in steps] == var


def test_scorer_picks_top_variant(monkeypatch):
    run_log = []
    mod_best = ModuleType("mod_best")
    mod_best.run = lambda: run_log.append("best") or True
    mod_worse = ModuleType("mod_worse")
    mod_worse.run = lambda: run_log.append("worse") or True
    sys.modules["mod_best"] = mod_best
    sys.modules["mod_worse"] = mod_worse

    generate_calls = []
    wem = _load_manager({"mod_best": 2.0, "mod_worse": 1.2}, generate_calls)

    def baseline():
        return True

    promoted = wem.evolve(baseline, workflow_id=1, variants=2)
    assert generate_calls == [1]
    run_log.clear()
    promoted()
    assert run_log == ["best"]


def test_workflow_marked_stable_without_improvement():
    mod_fail = ModuleType("mod_fail")
    mod_fail.run = lambda: True
    sys.modules["mod_fail"] = mod_fail

    generate_calls = []
    wem = _load_manager({"mod_fail": 0.8}, generate_calls)

    def baseline():
        return True

    wem.evolve(baseline, workflow_id=2, variants=1)
    assert wem.is_stable(2)
    assert generate_calls == [2]


def test_diminishing_gates_repeat_evolution():
    mod_fail = ModuleType("mod_fail")
    mod_fail.run = lambda: True
    sys.modules["mod_fail"] = mod_fail

    generate_calls = []
    wem = _load_manager({"mod_fail": 0.8}, generate_calls)

    def baseline():
        return True

    wem.evolve(baseline, workflow_id=3, variants=1)
    assert generate_calls == [3]
    generate_calls.clear()
    wem.evolve(baseline, workflow_id=3, variants=1)
    assert generate_calls == []
