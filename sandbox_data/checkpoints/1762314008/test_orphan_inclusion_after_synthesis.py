import os
import shutil
import sys
import types
from pathlib import Path

import workflow_synthesizer as ws
from dynamic_path_router import resolve_path


def _copy_modules(tmp_path: Path) -> None:
    for mod in resolve_path("tests/fixtures/workflow_modules").glob("*.py"):  # path-ignore
        shutil.copy(mod, tmp_path / mod.name)


def test_orphan_inclusion_after_synthesis(monkeypatch, tmp_path):
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
    analytics_mod = types.ModuleType("analytics")
    analytics_mod.adaptive_roi_model = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "analytics", analytics_mod)
    ar_mod = types.ModuleType("adaptive_roi_predictor")
    ar_mod.load_training_data = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "adaptive_roi_predictor", ar_mod)

    _copy_modules(tmp_path)
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    (extra_dir / "mod.py").write_text("VALUE = 1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    od_mod = types.ModuleType("sandbox_runner.orphan_discovery")
    od_mod.discover_recursive_orphans = lambda repo: {"extra.mod": []}
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_discovery", od_mod)

    auto_called: dict[str, list[str]] = {}

    def fake_auto(paths, recursive=True, router=None, context_builder=None):
        auto_called["auto"] = list(paths)
        return None, {"added": list(paths)}

    def fake_try(mods, router=None, context_builder=None):
        auto_called["workflow"] = list(mods)

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.auto_include_modules = fake_auto
    env_mod.try_integrate_into_workflows = fake_try
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    class DummyGrapher:
        def __init__(self, root):
            self.graph = {}

        def build_graph(self, repo):
            return {}

        def update_graph(self, names):
            pass

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = DummyGrapher
    mg_mod.load_graph = lambda p: {}
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    class DummyClusterer:
        def __init__(self):
            pass

        def index_modules(self, paths):
            pass

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyClusterer
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    def post_round_orphan_scan(repo, modules=None, *, logger=None, router=None):
        from sandbox_runner.environment import (
            auto_include_modules,
            try_integrate_into_workflows,
        )

        _, res = auto_include_modules(
            ["extra/mod.py"], recursive=True, router=router, context_builder=None
        )  # path-ignore
        added = res.get("added", [])
        if added:
            try_integrate_into_workflows(
                sorted(added), router=router, context_builder=None
            )
        return added, True, True

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    pkg.post_round_orphan_scan = post_round_orphan_scan
    pkg.try_integrate_into_workflows = fake_try
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)

    monkeypatch.setitem(
        sys.modules, "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None)
    )

    synth = ws.WorkflowSynthesizer()
    synth.generate_workflows(start_module="mod_a", limit=1, max_depth=1)

    assert auto_called["auto"] == ["extra/mod.py"]  # path-ignore
    assert auto_called["workflow"] == ["extra/mod.py"]  # path-ignore
