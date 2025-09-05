import sys
import types

# Stub external dependencies to keep the test lightweight
stub = types.ModuleType("stub")
for mod in (
    "psutil",
    "networkx",
    "pandas",
    "pulp",
    "scipy",
    "git",
):
    if mod not in sys.modules:
        sys.modules[mod] = stub
# provide attributes used by imports
stub.stats = stub
stub.ttest_ind_from_stats = lambda *a, **k: (0.0, 1.0)
stub.Repo = object
stub.DiGraph = object
stub.log2 = lambda *a, **k: 0.0

git_exc = types.ModuleType("git.exc")
git_exc.GitCommandError = Exception
git_exc.InvalidGitRepositoryError = Exception
git_exc.NoSuchPathError = Exception
sys.modules["git.exc"] = git_exc

# Optional visualization and config libraries
matplotlib_stub = types.ModuleType("matplotlib")
plt_stub = types.ModuleType("pyplot")
matplotlib_stub.pyplot = plt_stub  # path-ignore
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", plt_stub)  # path-ignore

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
dotenv_stub.dotenv_values = lambda *a, **k: {}
sys.modules.setdefault("dotenv", dotenv_stub)

prom_stub = types.ModuleType("prometheus_client")
prom_stub.CollectorRegistry = object
prom_stub.Counter = object
prom_stub.Gauge = object
sys.modules.setdefault("prometheus_client", prom_stub)

# Lightweight sklearn stubs
sys.modules.setdefault("sklearn", stub)
sys.modules.setdefault("sklearn.feature_extraction", stub)
sys.modules.setdefault("sklearn.feature_extraction.text", stub)
sys.modules.setdefault("sklearn.cluster", stub)
sys.modules.setdefault("sklearn.linear_model", stub)
stub.TfidfVectorizer = object
stub.KMeans = object
stub.LinearRegression = type("LinearRegression", (), {"__init__": lambda self, *a, **k: None})
sys.modules.setdefault("sklearn.model_selection", stub)
stub.train_test_split = lambda *a, **k: ([], [])
sys.modules.setdefault("sklearn.metrics", stub)
stub.accuracy_score = lambda *a, **k: 0.0
stub.LogisticRegression = type("LogisticRegression", (), {"__init__": lambda self, *a, **k: None})
sys.modules.setdefault("sklearn.ensemble", stub)
stub.RandomForestClassifier = type("RandomForestClassifier", (), {"__init__": lambda self, *a, **k: None})

from menace.mutation_lineage import MutationLineage
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from menace.code_database import PatchHistoryDB, PatchRecord
from menace.variant_manager import VariantManager
from menace.evolution_orchestrator import EvolutionOrchestrator


def test_lineage_variant_and_orchestrator_logging(monkeypatch, tmp_path):
    # setup databases with root and failing patches
    e_db = EvolutionHistoryDB(tmp_path / "e.db")
    p_db = PatchHistoryDB(tmp_path / "p.db")

    root_patch = PatchRecord(filename="wf.py", description="root", roi_before=0, roi_after=1)  # path-ignore
    root_id = p_db.add(root_patch)
    root_event = EvolutionEvent("root", 0, 1, 1.0, patch_id=root_id, workflow_id=1)
    root_event_id = e_db.add(root_event)

    bad_patch = PatchRecord(
        filename="wf.py",  # path-ignore
        description="bad",
        roi_before=1,
        roi_after=0.5,
        roi_delta=-0.5,
        parent_patch_id=root_id,
    )
    bad_id = p_db.add(bad_patch)
    bad_event = EvolutionEvent(
        "bad",
        1,
        0.5,
        -0.5,
        patch_id=bad_id,
        workflow_id=1,
        parent_event_id=root_event_id,
    )
    e_db.add(bad_event)

    ml = MutationLineage(history_db=e_db, patch_db=p_db)
    path = ml.backtrack_failed_path(bad_id)
    assert path == [bad_id, root_id]

    clone_id = ml.clone_branch_for_ab_test(root_id, "clone")
    with p_db._connect() as conn:  # type: ignore[attr-defined]
        parent = conn.execute("SELECT parent_patch_id FROM patch_history WHERE id=?", (clone_id,)).fetchone()[0]
    assert parent == root_id

    events = []

    def mock_log_mutation(
        change,
        reason,
        trigger,
        performance,
        workflow_id,
        before_metric=0.0,
        after_metric=0.0,
        parent_id=None,
    ):
        event_id = len(events) + 1
        events.append(
            {
                "change": change,
                "reason": reason,
                "trigger": trigger,
                "performance": performance,
                "before": before_metric,
                "after": after_metric,
                "parent_id": parent_id,
                "event_id": event_id,
            }
        )
        return event_id

    monkeypatch.setattr("menace.variant_manager.MutationLogger.log_mutation", mock_log_mutation)
    monkeypatch.setattr("menace.evolution_orchestrator.MutationLogger.log_mutation", mock_log_mutation)

    exp_mgr = types.SimpleNamespace()
    vm = VariantManager(exp_mgr, history_db=e_db)
    vm.spawn_variant(root_event_id, "expA")
    assert events[0]["change"] == "expA"
    assert events[0]["reason"] == "spawn_variant"
    assert events[0]["trigger"] == "variant_manager"
    assert events[0]["parent_id"] == root_event_id

    # orchestrator setup
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        log_evolution_cycle=lambda *a, **k: None,
    )
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 0.5)
    improver = types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(roi=None))
    evolver = types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(ga_results={}, predictions=[]))
    analysis = types.SimpleNamespace(predict=lambda action, metric: 0.0, train=lambda: None)
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=e_db,
        analysis_bot=analysis,
    )
    roi_vals = iter([1.0, 1.5, 2.0, 2.5])
    monkeypatch.setattr(orch, "_latest_roi", lambda: next(roi_vals))
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.5)
    orch.prev_roi = 1.0

    orch.run_cycle()
    first_id = e_db.conn.execute("SELECT max(rowid) FROM evolution_history").fetchone()[0]
    orch.run_cycle()
    second_id = e_db.conn.execute("SELECT max(rowid) FROM evolution_history").fetchone()[0]
    parent_id = e_db.conn.execute(
        "SELECT parent_event_id FROM evolution_history WHERE rowid=?", (second_id,)
    ).fetchone()[0]
    assert parent_id == first_id

    # last logged mutation corresponds to second cycle
    assert events[-1]["parent_id"] == second_id
    assert events[-1]["performance"] == 0.5
