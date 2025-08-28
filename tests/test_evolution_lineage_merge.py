import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Setup package context and stub heavy dependencies before import
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
pkg = types.ModuleType("menace_sandbox")
pkg.__path__ = [str(ROOT / "menace_sandbox")]
sys.modules.setdefault("menace_sandbox", pkg)
sys.modules.pop("menace_sandbox.workflow_evolution_manager", None)


def _stub(name: str, **attrs):
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
            "is_stable": lambda self, wid, *a, **k: False,
            "mark_stable": lambda self, *a, **k: None,
            "clear": lambda self, *a, **k: None,
            "get_ema": lambda self, wid: (0.0, 0),
            "set_ema": lambda self, wid, ema, count: None,
        },
    ),
)
_stub("evolution_history_db", EvolutionHistoryDB=object, EvolutionEvent=object)
_stub("mutation_logger", log_mutation=lambda **kw: 1, log_workflow_evolution=lambda **kw: None)
_stub("workflow_summary_db", WorkflowSummaryDB=object)
_stub(
    "sandbox_settings",
    SandboxSettings=lambda: SimpleNamespace(
        roi_ema_alpha=0.1,
        workflow_merge_similarity=0.9,
        workflow_merge_entropy_delta=0.1,
        duplicate_similarity=0.9,
        duplicate_entropy=0.1,
    ),
)
_stub("workflow_synergy_comparator", WorkflowSynergyComparator=object)
_stub("workflow_metrics", compute_workflow_entropy=lambda spec: 0.0)
_stub("workflow_benchmark", benchmark_workflow=lambda *a, **k: None)
_stub("workflow_merger", merge_workflows=lambda *a, **k: Path("merged.json"))
_stub(
    "workflow_run_summary",
    record_run=lambda *a, **k: None,
    save_all_summaries=lambda *a, **k: None,
)
_stub(
    "sandbox_runner",
    WorkflowSandboxRunner=type(
        "Runner",
        (),
        {"run": lambda self, fn, safe_mode=True: SimpleNamespace(modules=[])},
    ),
)
_stub(
    "workflow_synthesizer",
    save_workflow=lambda *a, **k: (Path("dummy.json"), {"workflow_id": "merged", "created_at": ""}),
)
_stub(
    "workflow_graph",
    WorkflowGraph=type(
        "Graph",
        (),
        {
            "add_workflow": lambda self, *a, **k: None,
            "add_dependency": lambda self, *a, **k: None,
        },
    ),
)
_stub("workflow_lineage", load_specs=lambda path: [])

import menace_sandbox.workflow_evolution_manager as wem  # noqa: E402

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
    "menace_sandbox.sandbox_settings",
    "menace_sandbox.workflow_synergy_comparator",
    "menace_sandbox.workflow_metrics",
    "menace_sandbox.workflow_merger",
    "menace_sandbox.workflow_run_summary",
    "menace_sandbox.sandbox_runner",
    "menace_sandbox.workflow_synthesizer",
    "menace_sandbox.workflow_graph",
    "menace_sandbox.workflow_lineage",
]:
    sys.modules.pop(mod, None)


def test_merge_similar_workflows(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    baseline_spec = [{"module": "a"}]
    variant_spec = [{"module": "a"}]

    class FakeBot:
        _rearranged_events: dict[str, int] = {}

        def generate_variants(self, limit, workflow_id):
            yield "a"

    monkeypatch.setattr(wem, "WorkflowEvolutionBot", lambda: FakeBot())

    run_calls: list[str] = []

    class FakeScorer:
        def __init__(self, results_db, tracker):
            pass

        def run(self, fn, wf_id, run_id):
            run_calls.append(run_id)
            if run_id == "baseline":
                roi, spec = 1.0, baseline_spec
            elif run_id.startswith("merge-"):
                roi, spec = 3.0, variant_spec
            else:
                roi, spec = 2.0, variant_spec
            return SimpleNamespace(
                roi_gain=roi, runtime=0.0, success_rate=1.0, workflow_spec=spec
            )

    monkeypatch.setattr(wem, "CompositeWorkflowScorer", FakeScorer)
    monkeypatch.setattr(
        wem,
        "ROIResultsDB",
        lambda: SimpleNamespace(log_module_delta=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        wem,
        "ROITracker",
        lambda: SimpleNamespace(
            calculate_raroi=lambda roi: (0, roi, 0),
            score_workflow=lambda wf, raroi: None,
            diminishing=lambda: 0,
        ),
    )
    monkeypatch.setattr(
        wem,
        "MutationLogger",
        SimpleNamespace(
            log_mutation=lambda **kw: 1,
            log_workflow_evolution=lambda **kw: None,
        ),
    )
    monkeypatch.setattr(wem, "EVOLUTION_DB", SimpleNamespace(add=lambda *a, **k: None))
    monkeypatch.setattr(wem, "EvolutionEvent", lambda *a, **k: None)
    monkeypatch.setattr(wem, "_update_ema", lambda *a, **k: False)
    monkeypatch.setattr(wem.STABLE_WORKFLOWS, "mark_stable", lambda *a, **k: None)
    monkeypatch.setattr(wem.STABLE_WORKFLOWS, "clear", lambda *a, **k: None)
    monkeypatch.setattr(wem.STABLE_WORKFLOWS, "is_stable", lambda wid, *a, **k: False)

    merge_called: dict[str, Any] = {"count": 0}

    class FakeComparator:
        @staticmethod
        def compare(a, b):
            return SimpleNamespace(aggregate=1.0, entropy_a=0.0, entropy_b=0.0)

        @classmethod
        def merge_duplicate(cls, base_id, dup_id, out_dir="workflows"):
            merge_called["args"] = (base_id, dup_id, out_dir)
            merge_called["count"] += 1
            out = Path(out_dir) / f"{base_id}.merged.json"
            out.write_text(
                json.dumps({"steps": variant_spec, "metadata": {"workflow_id": "merged"}})
            )
            return out

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", FakeComparator)

    result_callable = wem.evolve(lambda: True, 1, variants=1)
    assert "args" in merge_called
    assert merge_called["count"] == 1
    assert sum(r.startswith("merge-") for r in run_calls) == 1
    assert getattr(result_callable, "workflow_id") == "merged"
    assert getattr(result_callable, "parent_id") == 1
