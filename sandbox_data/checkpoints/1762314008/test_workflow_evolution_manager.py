from types import SimpleNamespace
import sys
from pathlib import Path
import types
import json

from dynamic_path_router import resolve_path

# Ensure package context and stub heavy dependencies before import
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
pkg = types.ModuleType("menace_sandbox")
pkg.__path__ = [str(ROOT / "menace_sandbox")]
sys.modules.setdefault("menace_sandbox", pkg)

def _load_steps(name: str) -> list[dict]:
    path = resolve_path(f"tests/fixtures/workflows/{name}")
    return json.loads(path.read_text()).get("steps", [])


def _stub(name, **attrs):
    mod = types.ModuleType(f"menace_sandbox.{name}")
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[f"menace_sandbox.{name}"] = mod

_stub("composite_workflow_scorer", CompositeWorkflowScorer=object)
_stub("workflow_evolution_bot", WorkflowEvolutionBot=object)
_stub("roi_results_db", ROIResultsDB=object)
_stub("roi_tracker", ROITracker=object)


class _FailingWorkflowStabilityDB:
    init_calls = 0

    def __init__(self, *args, **kwargs):
        type(self).init_calls += 1
        raise RuntimeError("stability db unavailable")


class _FailingEvolutionHistoryDB:
    init_calls = 0

    def __init__(self, *args, **kwargs):
        type(self).init_calls += 1
        raise RuntimeError("evolution db unavailable")


_stub("workflow_stability_db", WorkflowStabilityDB=_FailingWorkflowStabilityDB)
_stub(
    "evolution_history_db",
    EvolutionHistoryDB=_FailingEvolutionHistoryDB,
    EvolutionEvent=None,
)
_stub(
    "mutation_logger",
    log_mutation=lambda **kw: 1,
    log_workflow_evolution=lambda **kw: None,
)
_stub("workflow_summary_db", WorkflowSummaryDB=object)
_stub(
    "sandbox_settings",
    SandboxSettings=lambda: SimpleNamespace(
        roi_ema_alpha=0.1,
        workflow_merge_similarity=0.9,
        workflow_merge_entropy_delta=0.1,
        duplicate_similarity=0.95,
        duplicate_entropy=0.05,
    ),
)
_stub("workflow_synergy_comparator", WorkflowSynergyComparator=object)
_stub("workflow_metrics", compute_workflow_entropy=lambda spec: 0.0)
_stub("workflow_merger", merge_workflows=lambda *a, **k: Path("merged.json"))

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
"menace_sandbox.workflow_graph",
]:
    sys.modules.pop(mod, None)


def test_import_does_not_initialize_databases():
    assert _FailingWorkflowStabilityDB.init_calls == 0
    assert _FailingEvolutionHistoryDB.init_calls == 0


def test_get_stability_db_lazy(monkeypatch):
    start = _FailingWorkflowStabilityDB.init_calls
    monkeypatch.setattr(wem, "_STABILITY_DB", None, raising=False)
    monkeypatch.setattr(wem, "_STABILITY_DB_INITIALIZED", False, raising=False)
    assert wem._get_stability_db() is None
    assert _FailingWorkflowStabilityDB.init_calls == start + 1


def test_get_evolution_db_lazy(monkeypatch):
    start = _FailingEvolutionHistoryDB.init_calls
    monkeypatch.setattr(wem, "_EVOLUTION_DB", None, raising=False)
    monkeypatch.setattr(wem, "_EVOLUTION_DB_INITIALIZED", False, raising=False)
    assert wem._get_evolution_db() is None
    assert _FailingEvolutionHistoryDB.init_calls == start + 1


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

    class FakeStabilityDB:
        def __init__(self):
            self._ema: dict[str, tuple[float, int]] = {}

        def get_ema(self, workflow_id: str) -> tuple[float, int]:
            return self._ema.get(workflow_id, (0.0, 0))

        def set_ema(self, workflow_id: str, ema: float, count: int) -> None:
            self._ema[workflow_id] = (ema, count)

        def mark_stable(self, *args, **kwargs):
            pass

        def clear(self, *args, **kwargs):
            pass

        def is_stable(self, *args, **kwargs) -> bool:
            return False

    fake_stability = FakeStabilityDB()
    fake_evolution = SimpleNamespace(add=lambda *a, **k: None)

    monkeypatch.setattr(wem, "_get_stability_db", lambda: fake_stability)
    monkeypatch.setattr(wem, "_TEST_STABILITY_DB", fake_stability, raising=False)
    monkeypatch.setattr(wem, "_get_evolution_db", lambda: fake_evolution)
    monkeypatch.setattr(wem, "_update_ema", lambda *a, **k: False)

    graph_called = {}

    class FakeGraph:

        def update_workflow(self, wid, roi=None, synergy_scores=None):
            graph_called["args"] = (wid, roi)

        def add_dependency(self, parent, child, **kwargs):
            graph_called["dependency"] = (parent, child, kwargs)
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

    class FakeRunner:
        def run(self, *args, **kwargs):
            return SimpleNamespace(modules=[])

    monkeypatch.setattr(
        wem.sandbox_runner, "WorkflowSandboxRunner", lambda *a, **k: FakeRunner()
    )

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

    baseline_spec = _load_steps("simple_ab.json")
    variant_spec = _load_steps("simple_ab.json")

    _setup(
        monkeypatch,
        baseline_roi=1.0,
        variant_roi=0.8,
        baseline_spec=baseline_spec,
        variant_spec=variant_spec,
        merge_roi=1.2,
        run_log=run_ids,
    )

    merge_called = {}

    class DummyComparator:
        @classmethod
        def compare(cls, a_spec, b_spec):
            return SimpleNamespace(
                aggregate=1.0,
                entropy_a=0.0,
                entropy_b=0.0,
            )

        @classmethod
        def merge_duplicate(cls, base_id, dup_id, out_dir="workflows"):
            merge_called["called"] = True
            out = Path(out_dir) / f"{base_id}.merged.json"
            out.write_text(
                json.dumps({"steps": variant_spec, "metadata": {"workflow_id": 2}})
            )
            return out

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", DummyComparator)

    wem.evolve(lambda: True, 1, variants=1)

    assert merge_called.get("called")
    assert any(r.startswith("merge-") for r in run_ids)


def test_overfit_variant_skips_merge(monkeypatch, tmp_path):
    """Variants flagged as overfitting are not merged in the evolution loop."""
    baseline_spec = _load_steps("simple_ab.json")
    variant_spec = _load_steps("simple_ab.json")

    _setup(
        monkeypatch,
        baseline_roi=1.0,
        variant_roi=2.0,
        baseline_spec=baseline_spec,
        variant_spec=variant_spec,
    )

    merge_called: dict[str, bool] = {}

    class DummyComparator:
        @classmethod
        def compare(cls, a_spec, b_spec):
            report = SimpleNamespace(is_overfitting=lambda: True)
            return SimpleNamespace(
                aggregate=1.0,
                entropy_a=0.0,
                entropy_b=0.0,
                overfit_a=report,
                overfit_b=report,
            )

        @staticmethod
        def is_duplicate(
            a, b=None, *, similarity_threshold=0.95, entropy_threshold=0.05
        ):
            return True

        @classmethod
        def merge_duplicate(cls, *a, **k):
            merge_called["called"] = True

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", DummyComparator)

    wem.evolve(lambda: True, 1, variants=1)

    assert "called" not in merge_called


def test_promoted_duplicate_triggers_merge(monkeypatch, tmp_path):
    run_ids: list[str] = []
    baseline_spec = _load_steps("simple_ab.json")
    variant_spec = _load_steps("simple_bc.json")

    _setup(
        monkeypatch,
        baseline_roi=1.0,
        variant_roi=2.0,
        baseline_spec=baseline_spec,
        variant_spec=variant_spec,
        merge_roi=2.5,
        run_log=run_ids,
    )

    # Mark candidate workflow 99 as stable
    monkeypatch.setattr(
        wem._TEST_STABILITY_DB,
        "is_stable",
        lambda wid, *a, **k: wid == "99",
    )

    # Create candidate spec file
    workdir = tmp_path / "workflows"
    workdir.mkdir()
    (workdir / "99.workflow.json").write_text(
        json.dumps({"steps": variant_spec, "metadata": {"workflow_id": 99}})
    )
    monkeypatch.chdir(tmp_path)

    def fake_load_specs(directory="workflows"):
        yield {"workflow_id": "99"}

    monkeypatch.setattr(wem, "_load_specs", fake_load_specs)

    merge_called: dict[str, bool] = {}

    class DummyComparator:
        @classmethod
        def compare(cls, a_spec, b_spec):
            return SimpleNamespace(aggregate=1.0, entropy_a=0.0, entropy_b=0.0)

        @staticmethod
        def is_duplicate(
            a, b=None, *, similarity_threshold=0.95, entropy_threshold=0.05
        ):
            return True

        @classmethod
        def merge_duplicate(cls, base_id, dup_id, out_dir="workflows"):
            merge_called["called"] = True
            out_path = Path(out_dir) / f"{base_id}.merged.json"
            out_path.write_text(
                json.dumps({"steps": variant_spec, "metadata": {"workflow_id": 123}})
            )
            return out_path

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", DummyComparator)

    wem.evolve(lambda: True, 1, variants=1)

    assert merge_called.get("called")
    assert any(r.startswith("merge-") for r in run_ids)


def test_best_practice_variant_collapses(monkeypatch, tmp_path):
    baseline_spec = _load_steps("simple_ab.json")
    variant_spec = _load_steps("simple_ab.json")

    monkeypatch.setattr(
        wem,
        "SandboxSettings",
        lambda: SimpleNamespace(
            roi_ema_alpha=0.1,
            workflow_merge_similarity=0.9,
            workflow_merge_entropy_delta=0.1,
            duplicate_similarity=0.95,
            duplicate_entropy=0.05,
            best_practice_match_threshold=0.9,
        ),
    )

    _setup(
        monkeypatch,
        baseline_roi=1.0,
        variant_roi=2.0,
        baseline_spec=baseline_spec,
        variant_spec=variant_spec,
    )

    bp_file = tmp_path / "best.json"

    class DummyComparator:
        workflow_dir = tmp_path
        best_practices_file = bp_file

        @classmethod
        def _extract_modules(cls, spec):
            return [s.get("module") for s in spec.get("steps", [])]

        @classmethod
        def _update_best_practices(cls, modules):
            bp_file.write_text(json.dumps({"sequences": [modules]}))

        @classmethod
        def compare(cls, a_spec, b_spec):
            return SimpleNamespace(
                aggregate=0.0,
                entropy_a=0.0,
                entropy_b=0.0,
                best_practice_match_a=(0.0, []),
                best_practice_match_b=(0.95, ["a", "b"]),
            )

        @classmethod
        def merge_duplicate(cls, *a, **k):
            raise AssertionError("merge should not be called")

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", DummyComparator)

    promoted = wem.evolve(lambda: True, 1, variants=1)

    assert getattr(promoted, "parent_id") == "best_practice"
    data = json.loads(bp_file.read_text())
    assert ["a", "b"] in data["sequences"]


def test_variant_matching_existing_best_practice_merges(monkeypatch, tmp_path):
    baseline_spec = _load_steps("simple_ab.json")
    variant_spec = _load_steps("simple_ab.json")

    monkeypatch.setattr(
        wem,
        "SandboxSettings",
        lambda: SimpleNamespace(
            roi_ema_alpha=0.1,
            workflow_merge_similarity=0.9,
            workflow_merge_entropy_delta=0.1,
            duplicate_similarity=0.95,
            duplicate_entropy=0.05,
            best_practice_match_threshold=0.9,
        ),
    )

    _setup(
        monkeypatch,
        baseline_roi=1.0,
        variant_roi=2.0,
        baseline_spec=baseline_spec,
        variant_spec=variant_spec,
    )

    bp_file = tmp_path / "best.json"
    bp_file.write_text(json.dumps({"sequences": [["a", "b"]]}))

    class DummyComparator:
        workflow_dir = tmp_path
        best_practices_file = bp_file

        @classmethod
        def compare(cls, a_spec, b_spec):
            return SimpleNamespace(
                aggregate=0.0,
                entropy_a=0.0,
                entropy_b=0.0,
                best_practice_match_a=(0.95, ["a", "b"]),
                best_practice_match_b=(0.95, ["a", "b"]),
            )

        @classmethod
        def merge_duplicate(cls, *a, **k):  # pragma: no cover - should not run
            raise AssertionError("merge should not be called")

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", DummyComparator)

    promoted = wem.evolve(lambda: True, 1, variants=1)

    assert getattr(promoted, "parent_id") == "best_practice"


def test_non_matching_variant_evolves_normally(monkeypatch, tmp_path):
    baseline_spec = _load_steps("simple_ab.json")
    variant_spec = _load_steps("simple_bc.json")

    monkeypatch.setattr(
        wem,
        "SandboxSettings",
        lambda: SimpleNamespace(
            roi_ema_alpha=0.1,
            workflow_merge_similarity=0.9,
            workflow_merge_entropy_delta=0.1,
            duplicate_similarity=0.95,
            duplicate_entropy=0.05,
            best_practice_match_threshold=0.9,
        ),
    )

    _setup(
        monkeypatch,
        baseline_roi=1.0,
        variant_roi=2.0,
        baseline_spec=baseline_spec,
        variant_spec=variant_spec,
    )

    bp_file = tmp_path / "best.json"
    bp_file.write_text(json.dumps({"sequences": [["a", "b"]]}))

    called: dict[str, bool] = {}

    class DummyComparator:
        workflow_dir = tmp_path
        best_practices_file = bp_file

        @classmethod
        def compare(cls, a_spec, b_spec):
            return SimpleNamespace(
                aggregate=0.0,
                entropy_a=0.0,
                entropy_b=0.0,
                best_practice_match_a=(0.1, ["a", "b"]),
                best_practice_match_b=(0.1, ["a", "b"]),
            )

        @classmethod
        def _update_best_practices(cls, modules):  # pragma: no cover
            called["updated"] = True

        @classmethod
        def merge_duplicate(cls, *a, **k):  # pragma: no cover - should not run
            raise AssertionError("merge should not be called")

        @classmethod
        def _extract_modules(cls, spec):
            return [s.get("module") for s in spec.get("steps", [])]

    monkeypatch.setattr(wem, "WorkflowSynergyComparator", DummyComparator)

    promoted = wem.evolve(lambda: True, 1, variants=1)

    assert getattr(promoted, "parent_id") == 1
    assert "updated" not in called
