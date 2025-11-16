import os
import sys
import types
import shutil
import importlib
from pathlib import Path

import workflow_synthesizer as ws
from dynamic_path_router import resolve_path


def _copy_modules(tmp_path: Path) -> None:
    """Copy fixture workflow modules into *tmp_path* for testing."""
    for mod in resolve_path("tests/fixtures/workflow_modules").glob("*.py"):  # path-ignore
        shutil.copy(mod, tmp_path / mod.name)


def _load_qfe(monkeypatch):
    """Load quick_fix_engine with heavy dependencies stubbed out."""
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("menace_sandbox")
    pkg.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "menace_sandbox", pkg)
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.error_cluster_predictor",
        types.SimpleNamespace(ErrorClusterPredictor=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.error_bot",
        types.SimpleNamespace(ErrorDB=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.self_coding_manager",
        types.SimpleNamespace(SelfCodingManager=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.knowledge_graph",
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
            HumanAlignmentAgent=type(
                "HA", (), {"evaluate_changes": lambda self, *a, **k: {}}
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "violation_logger",
        types.SimpleNamespace(log_violation=lambda *a, **k: None),
    )
    return importlib.import_module("menace_sandbox.quick_fix_engine")


def test_generate_workflows_calls_post_round_scan(tmp_path, monkeypatch):
    """generate_workflows should invoke orphan integration helper."""
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    called = {"integrate": False}

    def fake_scan(repo, modules=None, *, logger=None, router=None):
        called["integrate"] = True
        return [], True, True

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    pkg.post_round_orphan_scan = fake_scan
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(
        sys.modules, "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None)
    )

    class SG:
        def __init__(self, root):
            self.graph = {}

        def build_graph(self, repo):
            return {}

        def update_graph(self, names):
            pass

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = SG
    mg_mod.load_graph = lambda p: {}
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = lambda *a, **k: types.SimpleNamespace(
        index_modules=lambda paths: None,
        _load_synergy_groups=lambda repo: {},
        _index_clusters=lambda groups: None,
    )
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    synth = ws.WorkflowSynthesizer()
    synth.generate_workflows(start_module="mod_a", limit=1, max_depth=1)

    assert called["integrate"] is True


def test_quick_fix_patch_cycle_indexes_orphans(tmp_path, monkeypatch):
    """Quick fix patch cycle should auto include and index new modules."""
    qfe = _load_qfe(monkeypatch)

    # Prepare repo with a module to patch and an orphan module
    (tmp_path / "foo.py").write_text("VALUE = 1\n")  # path-ignore
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    (extra_dir / "mod.py").write_text("VALUE = 1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    # Stub orphan integration utilities
    auto_called: dict[str, list[str]] = {}

    def fake_auto(paths, recursive=True, router=None, context_builder=None):
        auto_called["mods"] = list(paths)
        return None, {"added": list(paths)}

    def fake_try(mods, router=None, context_builder=None):
        auto_called["workflow"] = list(mods)

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.auto_include_modules = fake_auto
    env_mod.try_integrate_into_workflows = fake_try
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    synergy_called: dict[str, list[str]] = {}

    class DummyGrapher:
        def __init__(self, root):
            self.graph = {}

        def update_graph(self, names):
            synergy_called["names"] = names

    intent_called: dict[str, list[Path]] = {}

    class DummyClusterer:
        def __init__(self, *a, **k):
            pass

        def index_modules(self, paths):
            intent_called["paths"] = list(paths)

        def _load_synergy_groups(self, repo):
            return {}

        def _index_clusters(self, groups):
            pass

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = DummyGrapher
    mg_mod.load_graph = lambda p: None
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyClusterer
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    def post_round_orphan_scan(repo, modules=None, *, logger=None, router=None):
        from sandbox_runner.environment import (
            auto_include_modules,
            try_integrate_into_workflows,
        )

        auto_include_modules(
            ["extra/mod.py"],
            recursive=True,
            router=router,
            context_builder=None,
        )  # path-ignore
        from module_synergy_grapher import ModuleSynergyGrapher
        ModuleSynergyGrapher(repo).update_graph(["extra.mod"])
        from intent_clusterer import IntentClusterer
        IntentClusterer(None, None).index_modules([Path(repo) / "extra/mod.py"])  # path-ignore
        try_integrate_into_workflows(
            ["extra/mod.py"], router=router, context_builder=None
        )  # path-ignore
        return ["extra/mod.py"], True, True  # path-ignore

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    pkg.post_round_orphan_scan = post_round_orphan_scan
    pkg.try_integrate_into_workflows = fake_try
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(
        sys.modules, "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None)
    )

    # Avoid heavy helpers inside generate_patch
    monkeypatch.setattr(qfe, "generate_code_diff", lambda *a, **k: {})
    monkeypatch.setattr(qfe, "flag_risky_changes", lambda *a, **k: [])
    monkeypatch.setattr(qfe, "_collect_diff_data", lambda *a, **k: {})
    monkeypatch.setattr(
        qfe,
        "HumanAlignmentAgent",
        lambda: types.SimpleNamespace(evaluate_changes=lambda *a, **k: {}),
    )
    monkeypatch.setattr(
        qfe,
        "EmbeddingBackfill",
        lambda: types.SimpleNamespace(run=lambda *a, **k: None),
    )

    class Engine:
        def generate_helper(self, desc, **kwargs):
            return "# patched\n"

        def apply_patch(self, path, helper, **kwargs):
            p = Path(path)
            p.write_text(p.read_text() + helper)
            return 1, "", ""

    eng = Engine()
    manager = types.SimpleNamespace(
        engine=eng, register_patch_cycle=lambda *a, **k: None
    )
    qfe.generate_patch(
        "foo.py",
        manager,
        eng,
        context_builder=qfe.ContextBuilder(),
        patch_logger=types.SimpleNamespace(track_contributors=lambda *a, **k: None),
    )  # path-ignore

    assert auto_called["mods"] == ["extra/mod.py"]  # path-ignore
    assert synergy_called["names"] == ["extra.mod"]
    assert intent_called["paths"] == [tmp_path / "extra/mod.py"]  # path-ignore
    assert auto_called["workflow"] == ["extra/mod.py"]  # path-ignore
