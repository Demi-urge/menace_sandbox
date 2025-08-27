import os
import sys
import types
import shutil
from pathlib import Path

from workflow_synthesizer import generate_variants

FIXTURES = Path(__file__).parent / "fixtures" / "workflow_modules"


def _copy_modules(tmp_path: Path) -> None:
    for name in ("mod_a.py", "mod_b.py"):
        shutil.copy(FIXTURES / name, tmp_path / name)


def test_generate_variants_integrates_orphans(monkeypatch, tmp_path):
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

    _copy_modules(tmp_path)
    (tmp_path / "helper.py").write_text("def helper(data):\n    return data\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    synergy_called: dict[str, list[str]] = {}
    intent_called: dict[str, list[Path]] = {}

    class DummyGrapher:
        def __init__(self, root: Path):
            self.graph = {}

        def build_graph(self, repo: Path) -> dict[str, list[str]]:
            return {}

        def update_graph(self, names: list[str]) -> None:
            synergy_called["names"] = list(names)

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = DummyGrapher
    mg_mod.load_graph = lambda p: {}
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    class DummyIntent:
        class Match:
            def __init__(self, path: str) -> None:
                self.path = path
                self.members = None

        def _search_related(self, prompt: str, top_k: int = 5):
            return [self.Match("helper.py")]

        def index_modules(self, paths):
            intent_called["paths"] = list(paths)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyIntent
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    integrate_called: dict[str, object] = {}
    workflow_called: dict[str, list[str]] = {}

    def integrate_new_orphans(repo: Path, router=None):
        integrate_called["called"] = True
        from module_synergy_grapher import ModuleSynergyGrapher
        ModuleSynergyGrapher(repo).update_graph(["helper"])
        from intent_clusterer import IntentClusterer
        IntentClusterer().index_modules([repo / "helper.py"])
        return ["helper.py"]

    def fake_try(mods, router=None):
        workflow_called["mods"] = list(mods)

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    pkg.integrate_new_orphans = integrate_new_orphans
    pkg.try_integrate_into_workflows = fake_try
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None))

    variants = generate_variants(["mod_a", "mod_b"], 5, None, DummyIntent())

    assert any("helper" in v for v in variants)
    assert integrate_called.get("called") is True
    assert workflow_called["mods"] == ["helper.py"]
    assert synergy_called["names"] == ["helper"]
    assert intent_called["paths"] == [tmp_path / "helper.py"]
