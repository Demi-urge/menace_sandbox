from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast
import importlib.util
import sys

from workflow_synthesizer import (
    ModuleIOAnalyzer,
    WorkflowSynthesizer,
    generate_variants,
)
from dynamic_path_router import resolve_path


def test_generate_variants_filters_invalid_permutations(tmp_path, monkeypatch):
    for src in [
        resolve_path("tests/fixtures/workflow_modules/mod_a.py"),  # path-ignore
        resolve_path("tests/fixtures/workflow_modules/mod_b.py"),  # path-ignore
        resolve_path("tests/fixtures/workflow_modules/mod_c.py"),  # path-ignore
    ]:
        (tmp_path / src.stem).write_text(src.read_text())
    monkeypatch.chdir(tmp_path)

    base = ["mod_a", "mod_b", "mod_c"]
    invalid_start = "mod_b"

    class Swapper:
        def get_synergy_cluster(self, module_name, threshold=0.7, bfs=False):
            if module_name == "mod_a":
                return {invalid_start}
            if module_name == "mod_b":
                return {"mod_a"}
            return {module_name}

    variants = generate_variants(base, 5, Swapper(), None)

    assert variants
    assert all(v[0] != invalid_start for v in variants)

    analyzer = ModuleIOAnalyzer()
    checker = WorkflowSynthesizer()
    for var in variants:
        modules = [analyzer.analyze(Path(m)) for m in var]
        steps = checker.resolve_dependencies(modules)
        assert not any(s.unresolved for s in steps)
        assert [s.module for s in steps] == var


def test_benchmark_workflow_variants_calculates_roi_delta():
    class ROIResultsDB:
        def log_module_delta(self, *a, **k):
            pass

    class CompositeWorkflowScorer:
        def __init__(self, *a, **k):
            pass

        def run(self, fn, wf_id, run_id):
            roi = 1.0 if run_id == "baseline" else 1.5
            return SimpleNamespace(roi_gain=roi, runtime=0.0, success_rate=1.0)

    class MutationLogger:
        @staticmethod
        def log_mutation(**kw):
            return 1

        @staticmethod
        def record_mutation_outcome(*a, **k):
            pass

    src = resolve_path("self_improvement/orchestration_utils.py").read_text()  # path-ignore
    tree = ast.parse(src)
    func_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "benchmark_workflow_variants"
    )
    module = ast.Module([func_node], type_ignores=[])
    ns: dict[str, object] = {
        "ROIResultsDB": ROIResultsDB,
        "CompositeWorkflowScorer": CompositeWorkflowScorer,
        "MutationLogger": MutationLogger,
        "EvaluationResult": SimpleNamespace,
    }
    exec(compile(module, filename="<ast>", mode="exec"), ns)
    bench = ns["benchmark_workflow_variants"]

    def baseline():
        return True

    def variant():
        return True

    results = bench(1, {"baseline": baseline, "winner": variant})
    assert results["baseline"][1] == 0.0
    assert results["winner"][1] == 0.5


def _import_wem(side_effects, generate_calls=None):
    pkg = ModuleType("menace_sandbox")
    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["menace_sandbox"] = pkg

    cws_mod = ModuleType("menace_sandbox.composite_workflow_scorer")

    class CompositeWorkflowScorer:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            roi_gain = side_effects.pop(0)
            return SimpleNamespace(roi_gain=roi_gain, runtime=0.0, success_rate=1.0)

    cws_mod.CompositeWorkflowScorer = CompositeWorkflowScorer
    sys.modules["menace_sandbox.composite_workflow_scorer"] = cws_mod

    bot_mod = ModuleType("menace_sandbox.workflow_evolution_bot")

    class WorkflowEvolutionBot:
        _rearranged_events = {}

        def generate_variants(self, limit, workflow_id):
            if generate_calls is not None:
                generate_calls.append(workflow_id)
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
    logged: list[dict] = []

    def log_mutation(**kw):
        logged.append(kw)
        return 1

    def record_mutation_outcome(*a, **k):
        pass

    def log_workflow_evolution(**kw):
        logged.append(kw)
        return 1

    mut_mod.log_mutation = log_mutation
    mut_mod.record_mutation_outcome = record_mutation_outcome
    mut_mod.log_workflow_evolution = log_workflow_evolution
    sys.modules["menace_sandbox.mutation_logger"] = mut_mod

    tracker_mod = ModuleType("menace_sandbox.roi_tracker")

    class ROITracker:
        def __init__(self, *a, **k):
            self.roi_history = []

        def diminishing(self):
            return 0.1

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

    stab_db = WorkflowStabilityDB()
    stab_mod.WorkflowStabilityDB = lambda *a, **k: stab_db
    sys.modules["menace_sandbox.workflow_stability_db"] = stab_mod

    summary_mod = ModuleType("menace_sandbox.workflow_summary_db")

    class WorkflowSummaryDB:
        def set_summary(self, *a, **k):
            pass
    summary_mod.WorkflowSummaryDB = WorkflowSummaryDB
    sys.modules["menace_sandbox.workflow_summary_db"] = summary_mod

    run_summary_mod = ModuleType("menace_sandbox.workflow_run_summary")
    run_summary_mod.record_run = lambda *a, **k: None
    run_summary_mod.save_all_summaries = lambda *a, **k: None
    sys.modules["menace_sandbox.workflow_run_summary"] = run_summary_mod

    benchmark_mod = ModuleType("menace_sandbox.workflow_benchmark")
    benchmark_mod.benchmark_workflow = lambda *a, **k: None
    sys.modules["menace_sandbox.workflow_benchmark"] = benchmark_mod

    data_bot_mod = ModuleType("menace_sandbox.data_bot")
    data_bot_mod.MetricsDB = object
    sys.modules["menace_sandbox.data_bot"] = data_bot_mod

    neuro_mod = ModuleType("menace_sandbox.neuroplasticity")
    neuro_mod.PathwayDB = object
    sys.modules["menace_sandbox.neuroplasticity"] = neuro_mod

    graph_mod = ModuleType("menace_sandbox.workflow_graph")
    graph_mod.WorkflowGraph = object
    sys.modules["menace_sandbox.workflow_graph"] = graph_mod

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.workflow_evolution_manager",
        resolve_path("workflow_evolution_manager.py"),  # path-ignore
    )
    wem = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox.workflow_evolution_manager"] = wem
    assert spec.loader is not None
    spec.loader.exec_module(wem)
    wem.STABLE_WORKFLOWS.clear_all()
    return wem, logged, stab_db


def test_evolve_promotes_variant_on_roi_gain():
    side_effects = [1.0, 2.0]
    wem, logged, _ = _import_wem(side_effects)

    wf_id = 1
    wem.evolve(lambda: True, wf_id, variants=1)

    assert any(r.get("reason") == "promoted" for r in logged)
    assert not wem.is_stable(wf_id)


def test_evolve_marks_stable_and_gates_repeats():
    side_effects = [1.0, 0.95, 1.0]
    generate_calls: list[int] = []
    wem, logged, _ = _import_wem(side_effects, generate_calls)

    wf_id = 2
    wem.evolve(lambda: True, wf_id, variants=1)
    assert any(r.get("reason") == "stable" for r in logged)
    assert wem.is_stable(wf_id)

    wem.evolve(lambda: True, wf_id, variants=1)
    assert generate_calls == [wf_id]
    assert wem.is_stable(wf_id)
