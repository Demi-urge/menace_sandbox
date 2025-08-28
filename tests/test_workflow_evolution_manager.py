from types import SimpleNamespace
import sys
from pathlib import Path
import types
import json

# Ensure package context and stub heavy dependencies before import
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
pkg = types.ModuleType("menace_sandbox")
pkg.__path__ = [str(ROOT / "menace_sandbox")]
sys.modules.setdefault("menace_sandbox", pkg)

def _stub(name, **attrs):
    mod = types.ModuleType(f"menace_sandbox.{name}")
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[f"menace_sandbox.{name}"] = mod

_stub("composite_workflow_scorer", CompositeWorkflowScorer=object)
_stub("workflow_evolution_bot", WorkflowEvolutionBot=object)
_stub("roi_results_db", ROIResultsDB=object)
_stub("roi_tracker", ROITracker=object)
_stub(
    "workflow_stability_db",
    WorkflowStabilityDB=type(
        "WorkflowStabilityDB",
        (),
        {
            "is_stable": lambda self, *a, **k: False,
            "mark_stable": lambda self, *a, **k: None,
            "clear": lambda self, *a, **k: None,
        },
    ),
)
_stub("evolution_history_db", EvolutionHistoryDB=object, EvolutionEvent=object)
_stub(
    "mutation_logger",
    log_mutation=lambda **kw: 1,
    log_workflow_evolution=lambda **kw: None,
)
_stub("workflow_summary_db", WorkflowSummaryDB=object)
_stub("sandbox_settings", SandboxSettings=lambda: SimpleNamespace(roi_ema_alpha=0.1))
_stub("workflow_synergy_comparator", WorkflowSynergyComparator=object)
_stub("workflow_metrics", compute_workflow_entropy=lambda spec: 0.0)
_stub("workflow_merger", merge_workflows=lambda *a, **k: Path("merged.json"))

import menace_sandbox.workflow_evolution_manager as wem

# Restore real modules for other tests
for mod in [
    "menace_sandbox.composite_workflow_scorer",
    "menace_sandbox.workflow_evolution_bot",
    "menace_sandbox.roi_results_db",
    "menace_sandbox.roi_tracker",
    "menace_sandbox.workflow_stability_db",
    "menace_sandbox.evolution_history_db",
    "menace_sandbox.mutation_logger",
    "menace_sandbox.workflow_summary_db",
    "menace_sandbox.workflow_graph",
]:
    sys.modules.pop(mod, None)

def _setup(
    monkeypatch,
    baseline_roi: float = 1.0,
    variant_roi: float = 2.0,
    variant: str = "b-a",
    *,
    baseline_spec=None,
    variant_spec=None,
    merge_roi: float | None = None,
    run_log: list[str] | None = None,
):
    class FakeBot:
        _rearranged_events: dict[str, int] = {}

        def generate_variants(self, limit, workflow_id):
            yield variant

    monkeypatch.setattr(wem, "WorkflowEvolutionBot", lambda: FakeBot())

    class FakeScorer:
        def __init__(self, results_db, tracker):
            pass

        def run(self, fn, wf_id, run_id):
            if run_log is not None:
                run_log.append(run_id)
            if run_id == "baseline":
                roi = baseline_roi
                spec = baseline_spec
            elif run_id.startswith("merge-"):
                roi = merge_roi if merge_roi is not None else variant_roi
                spec = variant_spec
            else:
                roi = variant_roi
                spec = variant_spec
            return SimpleNamespace(
                roi_gain=roi,
                runtime=0.0,
                success_rate=1.0,
                workflow_spec=spec,
            )

    monkeypatch.setattr(wem, "CompositeWorkflowScorer", FakeScorer)

    class FakeResultsDB:
        def log_module_delta(self, *a, **k):
            pass
    monkeypatch.setattr(wem, "ROIResultsDB", lambda: FakeResultsDB())

    class FakeTracker:
        def calculate_raroi(self, roi):
            return 0, roi, 0
        def score_workflow(self, wf, raroi):
            pass
        def diminishing(self):
            return 0
    monkeypatch.setattr(wem, "ROITracker", lambda: FakeTracker())

    monkeypatch.setattr(
        wem,
        "MutationLogger",
        SimpleNamespace(log_mutation=lambda **kw: 1, log_workflow_evolution=lambda **kw: None),
    )

    monkeypatch.setattr(wem.STABLE_WORKFLOWS, "mark_stable", lambda *a, **k: None)
    monkeypatch.setattr(wem.STABLE_WORKFLOWS, "clear", lambda *a, **k: None)
    monkeypatch.setattr(wem.STABLE_WORKFLOWS, "is_stable", lambda *a, **k: False)
    monkeypatch.setattr(wem, "_update_ema", lambda *a, **k: False)

    graph_called = {}
    class FakeGraph:
        def update_workflow(self, wid, roi=None, synergy_scores=None):
            graph_called["args"] = (wid, roi)
    monkeypatch.setattr(wem, "WorkflowGraph", lambda *a, **k: FakeGraph())

    summary_called = {}
    class FakeSummaryDB:
        def set_summary(self, wid, summary):
            summary_called["args"] = (wid, summary)
    monkeypatch.setattr(wem, "WorkflowSummaryDB", lambda *a, **k: FakeSummaryDB())

    # Stub optional modules imported during promotion
    fake_bench = types.ModuleType("menace_sandbox.workflow_benchmark")
    fake_bench.benchmark_workflow = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.workflow_benchmark", fake_bench)
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.data_bot",
        types.ModuleType("menace_sandbox.data_bot"),
    )
    sys.modules["menace_sandbox.data_bot"].MetricsDB = lambda *a, **k: None
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.neuroplasticity",
        types.ModuleType("menace_sandbox.neuroplasticity"),
    )
    sys.modules["menace_sandbox.neuroplasticity"].PathwayDB = lambda *a, **k: None

    return graph_called, summary_called


def test_variant_promotion_updates_graph(monkeypatch):
    graph_called, summary_called = _setup(monkeypatch, baseline_roi=1.0, variant_roi=2.0)
    wem.evolve(lambda: True, 1, variants=1)
    assert graph_called["args"] == ("1", 2.0)
    assert "args" not in summary_called


def test_no_improvement_marks_stable(monkeypatch):
    graph_called, summary_called = _setup(monkeypatch, baseline_roi=1.0, variant_roi=0.5)
    wem.evolve(lambda: True, 1, variants=1)
    assert summary_called["args"][0] == 1
    assert "stable" in summary_called["args"][1]
    assert "args" not in graph_called


def test_near_identical_low_entropy_workflows_are_merged(monkeypatch, tmp_path):
    run_ids: list[str] = []

    baseline_spec = [{"module": "a", "outputs": ["x"]}]
    variant_spec = [{"module": "a", "outputs": ["x"]}]

    _setup(
        monkeypatch,
        baseline_roi=1.0,
        variant_roi=0.8,
        baseline_spec=baseline_spec,
        variant_spec=variant_spec,
        merge_roi=1.2,
        run_log=run_ids,
    )

    class DummyComparator:
        @classmethod
        def compare(cls, a_spec, b_spec):
            return SimpleNamespace(efficiency=1.0, modularity=1.0, expandability=0.0)

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", DummyComparator)

    merge_called = {}

    def fake_merge(base_path, a_path, b_path, out_path):
        merge_called["called"] = True
        out_path.write_text(
            json.dumps({"steps": variant_spec, "metadata": {"workflow_id": 2}})
        )
        return out_path

    monkeypatch.setattr(wem.workflow_merger, "merge_workflows", fake_merge)

    wem.evolve(lambda: True, 1, variants=1)

    assert merge_called.get("called")
    assert any(r.startswith("merge-") for r in run_ids)
