import os
import sys
import types
import shutil
from pathlib import Path

from workflow_synthesizer import generate_variants
from dynamic_path_router import resolve_path


def _copy_modules(tmp_path: Path) -> None:
    for name in ("mod_a.py", "mod_b.py"):  # path-ignore
        shutil.copy(
            resolve_path(f"tests/fixtures/workflow_modules/{name}"),
            tmp_path / name,
        )


def test_generate_variants_integrates_orphans(monkeypatch, tmp_path):
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

    _copy_modules(tmp_path)
    (tmp_path / "helper.py").write_text("def helper(data):\n    return data\n")  # path-ignore

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
            return [self.Match("helper.py")]  # path-ignore

        def index_modules(self, paths):
            intent_called["paths"] = list(paths)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyIntent
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    integrate_called: dict[str, object] = {}
    workflow_called: dict[str, list[str]] = {}

    def post_round_orphan_scan(repo: Path, modules=None, *, logger=None, router=None):
        integrate_called["called"] = True
        from module_synergy_grapher import ModuleSynergyGrapher
        ModuleSynergyGrapher(repo).update_graph(["helper"])
        from intent_clusterer import IntentClusterer
        IntentClusterer().index_modules([repo / "helper.py"])  # path-ignore
        workflow_called["mods"] = ["helper.py"]  # path-ignore
        return ["helper.py"], True, True  # path-ignore

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    pkg.post_round_orphan_scan = post_round_orphan_scan
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "db_router", types.SimpleNamespace(GLOBAL_ROUTER=None))

    variants = generate_variants(["mod_a", "mod_b"], 5, None, DummyIntent())

    assert any("helper" in v for v in variants)
    assert integrate_called.get("called") is True
    assert workflow_called["mods"] == ["helper.py"]  # path-ignore
    assert synergy_called["names"] == ["helper"]
    assert intent_called["paths"] == [tmp_path / "helper.py"]  # path-ignore
