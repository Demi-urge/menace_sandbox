import sys
import types
import importlib
from pathlib import Path


# --- Helpers ---------------------------------------------------------------

def _load_qfe(monkeypatch):
    """Load quick_fix_engine with heavy deps stubbed."""
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setitem(
        sys.modules,
        "menace.error_cluster_predictor",
        types.SimpleNamespace(ErrorClusterPredictor=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.error_bot",
        types.SimpleNamespace(ErrorDB=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.self_coding_manager",
        types.SimpleNamespace(SelfCodingManager=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.knowledge_graph",
        types.SimpleNamespace(KnowledgeGraph=object),
    )
    class DummyBuilder:
        def __init__(self, *a, **k):
            pass

        def refresh_db_weights(self):
            return None

        def build(self, *a, **k):
            return ""

    monkeypatch.setitem(
        sys.modules,
        "vector_service",
        types.SimpleNamespace(
            ContextBuilder=DummyBuilder,
            Retriever=object,
            FallbackResult=object,
            EmbeddingBackfill=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "patch_provenance",
        types.SimpleNamespace(PatchLogger=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "human_alignment_flagger",
        types.SimpleNamespace(_collect_diff_data=lambda *a, **k: {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "human_alignment_agent",
        types.SimpleNamespace(
            HumanAlignmentAgent=type("HA", (), {"evaluate_changes": lambda self, *a, **k: {}})
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "violation_logger",
        types.SimpleNamespace(log_violation=lambda *a, **k: None),
    )
    spec = importlib.util.spec_from_file_location(
        "menace.quick_fix_engine", root / "quick_fix_engine.py"  # path-ignore
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules["menace.quick_fix_engine"] = module
    return module


# --- QuickFixEngine tests --------------------------------------------------

def test_generate_patch_triggers_scan_once_success(monkeypatch, tmp_path):
    qfe = _load_qfe(monkeypatch)
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("VALUE = 1\n")
    monkeypatch.chdir(tmp_path)

    calls = []

    def fake_scan(repo, modules=None, *, logger=None, router=None):
        calls.append(repo)
        return [], True, True

    pkg = types.ModuleType("sandbox_runner")
    pkg.post_round_orphan_scan = fake_scan
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)

    class Engine:
        def generate_helper(self, desc, **kwargs):
            return "code"

        def apply_patch(self, p, helper, **k):
            return 1, "", ""

    monkeypatch.setattr(qfe, "generate_code_diff", lambda *a, **k: [])
    monkeypatch.setattr(qfe, "flag_risky_changes", lambda *a, **k: [])

    res = qfe.generate_patch(
        str(path), engine=Engine(), context_builder=qfe.ContextBuilder()
    )
    assert res == 1
    assert len(calls) == 1


def test_generate_patch_scan_failure_handled(monkeypatch, tmp_path):
    qfe = _load_qfe(monkeypatch)
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("VALUE = 1\n")
    monkeypatch.chdir(tmp_path)

    calls = []

    def fake_scan(repo, modules=None, *, logger=None, router=None):
        calls.append(repo)
        raise RuntimeError("boom")

    pkg = types.ModuleType("sandbox_runner")
    pkg.post_round_orphan_scan = fake_scan
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)

    class Engine:
        def generate_helper(self, desc, **kwargs):
            return "code"

        def apply_patch(self, p, helper, **k):
            return 1, "", ""

    monkeypatch.setattr(qfe, "generate_code_diff", lambda *a, **k: [])
    monkeypatch.setattr(qfe, "flag_risky_changes", lambda *a, **k: [])

    res = qfe.generate_patch(
        str(path), engine=Engine(), context_builder=qfe.ContextBuilder()
    )
    assert res == 1
    assert len(calls) == 1


# --- SelfDebuggerSandbox tests --------------------------------------------
# Provide a lightweight sandbox_runner before importing the heavy test module
sandbox_stub = types.ModuleType("sandbox_runner")
sandbox_stub.__path__ = [str(Path(__file__).resolve().parents[1] / "sandbox_runner")]
sandbox_stub.post_round_orphan_scan = lambda *a, **k: ([], True, True)
sys.modules.setdefault("sandbox_runner", sandbox_stub)

from tests.test_self_debugger_sandbox import (
    sds,
    DummyTelem,
    DummyEngine,
    DummyBuilder,
)  # noqa: E402


def test_preemptive_fix_triggers_scan_once_success(monkeypatch, tmp_path):
    (tmp_path / "mod.py").write_text("VALUE = 1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)

    calls = []

    def fake_scan(repo, modules=None, *, logger=None, router=None):
        calls.append(repo)
        return [], True, True

    monkeypatch.setattr(sds, "post_round_orphan_scan", fake_scan)
    monkeypatch.setattr(sds, "generate_patch", lambda m, e, **k: 1)
    monkeypatch.setattr(sds, "_collect_diff_data", lambda *a, **k: {})

    predictor = types.SimpleNamespace(
        predict_high_risk_modules=lambda top_n=5: ["mod.py"]  # path-ignore
    )
    sandbox = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        error_predictor=predictor,
    )

    sandbox.preemptive_fix_high_risk_modules(limit=1)
    assert len(calls) == 1


def test_preemptive_fix_scan_failure_handled(monkeypatch, tmp_path):
    (tmp_path / "mod.py").write_text("VALUE = 1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)

    calls = []

    def fake_scan(repo, modules=None, *, logger=None, router=None):
        calls.append(repo)
        raise RuntimeError("boom")

    monkeypatch.setattr(sds, "post_round_orphan_scan", fake_scan)
    monkeypatch.setattr(sds, "generate_patch", lambda m, e, **k: 1)
    monkeypatch.setattr(sds, "_collect_diff_data", lambda *a, **k: {})

    predictor = types.SimpleNamespace(
        predict_high_risk_modules=lambda top_n=5: ["mod.py"]  # path-ignore
    )
    sandbox = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        error_predictor=predictor,
    )

    sandbox.preemptive_fix_high_risk_modules(limit=1)
    assert len(calls) == 1


# --- Integration test ------------------------------------------------------

def test_new_module_included_after_scan(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "existing.py").write_text("X = 1\n")  # path-ignore
    (repo / "new_mod.py").write_text("Y = 2\n")  # path-ignore

    workflow_list: list[str] = []

    def auto_include_modules(paths, recursive=True, router=None, context_builder=None):
        return None, {"added": list(paths)}

    def try_integrate_into_workflows(mods, router=None, context_builder=None):
        workflow_list.extend(mods)

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.auto_include_modules = auto_include_modules
    env_mod.try_integrate_into_workflows = try_integrate_into_workflows
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = lambda root: types.SimpleNamespace(
        graph={},
        update_graph=lambda names: None,
        build_graph=lambda repo: {},
    )
    mg_mod.load_graph = lambda p: {}
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = lambda: types.SimpleNamespace(
        index_modules=lambda paths: None
    )
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    from sandbox_runner.orphan_integration import post_round_orphan_scan

    added, syn_ok, intent_ok = post_round_orphan_scan(
        repo, modules=["new_mod.py"], logger=None, router=None  # path-ignore
    )

    assert added == [str(repo / "new_mod.py")]  # path-ignore
    assert workflow_list == [str(repo / "new_mod.py")]  # path-ignore
    assert syn_ok is True and intent_ok is True


def test_post_round_scan_discovery_failure(monkeypatch, tmp_path):
    repo = tmp_path

    od_mod = types.ModuleType("sandbox_runner.orphan_discovery")
    od_mod.discover_recursive_orphans = lambda repo: (_ for _ in ()).throw(RuntimeError("fail"))
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_discovery", od_mod)

    from sandbox_runner.orphan_integration import post_round_orphan_scan

    added, syn_ok, intent_ok = post_round_orphan_scan(repo)
    assert added == []
    assert syn_ok is False and intent_ok is False
