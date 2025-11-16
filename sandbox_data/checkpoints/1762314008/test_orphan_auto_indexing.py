import sys
import types
import shutil
from pathlib import Path
from types import SimpleNamespace

import workflow_synthesizer as ws
from dynamic_path_router import resolve_path


def _copy_modules(tmp_path: Path) -> None:
    """Copy fixture workflow modules into *tmp_path* for testing."""
    for mod in resolve_path("tests/fixtures/workflow_modules").glob("*.py"):  # path-ignore
        shutil.copy(mod, tmp_path / mod.name)


def test_generate_workflows_indexes_discovered_modules(tmp_path, monkeypatch):
    """Ensure newly discovered modules trigger synergy graph and intent indexing."""
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    # Stub orphan integration via centralized helper and record indexing calls
    calls: dict[str, list] = {}

    def fake_scan(repo, modules=None, *, logger=None, router=None):
        calls["synergy"] = ["extra.mod"]
        calls["intent"] = [Path(repo) / "extra/mod.py"]  # path-ignore
        calls["workflow"] = ["extra/mod.py"]  # path-ignore
        return ["extra/mod.py"], True, True  # path-ignore

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    pkg.post_round_orphan_scan = fake_scan
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "db_router", SimpleNamespace(GLOBAL_ROUTER=None))

    synth = ws.WorkflowSynthesizer()
    synth.generate_workflows(start_module="mod_a", limit=1, max_depth=1)

    assert calls["synergy"] == ["extra.mod"]
    assert calls["intent"] == [tmp_path / "extra/mod.py"]  # path-ignore


def _load_env():
    """Load sandbox_runner.environment with minimal dependencies."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
    pkg.__path__ = [str(root / "sandbox_runner")]
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.environment", root / "sandbox_runner" / "environment.py"  # path-ignore
    )
    env = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner.environment"] = env
    assert spec and spec.loader
    spec.loader.exec_module(env)  # type: ignore[attr-defined]
    return env


def _load_thb():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        root / "task_handoff_bot.py",  # path-ignore
        submodule_search_locations=[str(root)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    assert spec and spec.loader
    spec.loader.exec_module(thb)  # type: ignore[attr-defined]
    return thb


def test_generate_workflows_for_modules_auto_indexes(tmp_path, monkeypatch):
    """generate_workflows_for_modules should auto include and index orphans."""
    env = _load_env()
    _load_thb()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    # Stub discovery and auto inclusion
    orphan_mod = types.ModuleType("sandbox_runner.orphan_discovery")
    orphan_mod.discover_recursive_orphans = lambda repo: {"extra.mod": []}
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_discovery", orphan_mod)

    auto_called: dict[str, list[str]] = {}
    def fake_auto(mods, recursive=True, router=None, context_builder=None):
        auto_called["mods"] = list(mods)
        return None, {"added": ["extra/mod.py"]}  # path-ignore
    monkeypatch.setattr(env, "auto_include_modules", fake_auto)

    synergy_called: dict[str, list[str]] = {}
    class FakeGrapher:
        def __init__(self, root):
            self.graph = object()
        def update_graph(self, names):
            synergy_called["names"] = names
    monkeypatch.setitem(
        sys.modules,
        "module_synergy_grapher",
        SimpleNamespace(ModuleSynergyGrapher=FakeGrapher, load_graph=lambda p: None),
    )

    intent_called: dict[str, list[Path]] = {}
    class FakeClusterer:
        def __init__(self, *a, **k):
            pass
        def index_modules(self, paths):
            intent_called["paths"] = list(paths)
        def _load_synergy_groups(self, repo):
            return {}
        def _index_clusters(self, groups):
            pass
    monkeypatch.setitem(
        sys.modules,
        "intent_clusterer",
        SimpleNamespace(IntentClusterer=FakeClusterer),
    )

    env.generate_workflows_for_modules(["foo.py"], workflows_db=tmp_path / "wf.db")  # path-ignore

    assert auto_called["mods"] == ["extra/mod.py"]  # path-ignore
    assert synergy_called["names"] == ["extra/mod"]
    assert intent_called["paths"] == [tmp_path / "extra/mod.py"]  # path-ignore


def test_evolve_auto_indexes_promoted_orphans(tmp_path, monkeypatch):
    """workflow_evolution_manager.evolve should index auto-included dependencies."""
    root = Path(__file__).resolve().parents[2]
    sys.path.append(str(root))
    pkg = types.ModuleType("menace_sandbox")
    pkg.__path__ = [str(root / "menace_sandbox")]
    sys.modules.setdefault("menace_sandbox", pkg)

    def _stub(name, **attrs):
        mod = types.ModuleType(f"menace_sandbox.{name}")
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[f"menace_sandbox.{name}"] = mod

    # Stub heavy dependencies for import
    _stub("composite_workflow_scorer", CompositeWorkflowScorer=object)
    _stub("workflow_evolution_bot", WorkflowEvolutionBot=object)
    _stub("roi_results_db", ROIResultsDB=object)
    _stub("roi_tracker", ROITracker=object)
    _stub(
        "workflow_stability_db",
        WorkflowStabilityDB=type(
            "WS", (), {
                "is_stable": lambda self, *a, **k: False,
                "mark_stable": lambda self, *a, **k: None,
                "clear": lambda self, *a, **k: None,
                "get_ema": lambda self, *a, **k: (0.0, 0),
                "set_ema": lambda self, *a, **k: None,
            }
        ),
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
    _stub("evolution_history_db", EvolutionHistoryDB=object, EvolutionEvent=object)
    _stub("workflow_graph", WorkflowGraph=object)
    _stub("mutation_logger", log_mutation=lambda **kw: 1, log_workflow_evolution=lambda **kw: None)

    # Prepare sandbox_runner package with orphan helpers
    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.__path__ = []
    auto_called: dict[str, list[str]] = {}
    sr_pkg.discover_recursive_orphans = lambda repo, module_map=None: {"extra.mod": []}
    sys.modules["sandbox_runner"] = sr_pkg

    env_mod = types.ModuleType("sandbox_runner.environment")
    def fake_auto(mods, recursive=True, validate=True, router=None, context_builder=None):
        auto_called["mods"] = list(mods)
        return None, {"added": ["extra/mod.py"]}  # path-ignore
    env_mod.auto_include_modules = fake_auto
    sys.modules["sandbox_runner.environment"] = env_mod

    def integrate_orphans(repo, router=None):
        from sandbox_runner.environment import auto_include_modules
        auto_include_modules(
            ["extra/mod.py"], recursive=True, router=router, context_builder=None
        )  # path-ignore
        from module_synergy_grapher import ModuleSynergyGrapher
        ModuleSynergyGrapher(repo).update_graph(["extra.mod"])
        from intent_clusterer import IntentClusterer
        IntentClusterer(None, None).index_modules([Path(repo) / "extra/mod.py"])  # path-ignore
        return ["extra/mod.py"]  # path-ignore

    post_mod = types.ModuleType("sandbox_runner.post_update")
    post_mod.integrate_orphans = integrate_orphans
    sys.modules["sandbox_runner.post_update"] = post_mod

    synergy_called: dict[str, list[str]] = {}
    class SG:
        def __init__(self, root):
            self.graph = object()
        def load(self, path):
            pass
        def build_graph(self, repo):
            pass
        def update_graph(self, names):
            synergy_called["names"] = names
    intent_called: dict[str, list[Path]] = {}
    class IC:
        def __init__(self, local_db_path, shared_db_path):
            pass
        def index_modules(self, paths):
            intent_called["paths"] = list(paths)
        def _load_synergy_groups(self, repo):
            return {}
        def _index_clusters(self, groups):
            pass
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", SimpleNamespace(ModuleSynergyGrapher=SG))
    monkeypatch.setitem(sys.modules, "intent_clusterer", SimpleNamespace(IntentClusterer=IC))

    # Import evolution manager with stubs
    sys.modules.pop("menace_sandbox.workflow_evolution_manager", None)
    import importlib
    import menace_sandbox.workflow_evolution_manager as wem
    importlib.reload(wem)

    # Replace stubbed classes with controllable fakes
    class FakeBot:
        _rearranged_events: dict[str, int] = {}
        def generate_variants(self, limit, workflow_id):
            yield "b-a"
    monkeypatch.setattr(wem, "WorkflowEvolutionBot", lambda: FakeBot())

    class FakeScorer:
        def __init__(self, results_db, tracker):
            pass
        def run(self, fn, wf_id, run_id):
            roi = 1.0 if run_id == "baseline" else 2.0
            return SimpleNamespace(roi_gain=roi, runtime=0.0, success_rate=1.0)
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
    monkeypatch.setattr(wem, "WorkflowGraph", lambda *a, **k: SimpleNamespace(update_workflow=lambda *a, **k: None))
    monkeypatch.setattr(wem, "WorkflowSummaryDB", lambda *a, **k: SimpleNamespace(set_summary=lambda *a, **k: None))

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path / "data"))

    wem.evolve(lambda: True, 1, variants=1)

    assert auto_called["mods"] == ["extra/mod.py"]  # path-ignore
    assert synergy_called["names"] == ["extra.mod"]
    assert intent_called["paths"] == [tmp_path / "extra/mod.py"]  # path-ignore
