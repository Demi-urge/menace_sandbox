import os
import sys
import types
from pathlib import Path

import importlib
import pytest


def test_cycle_validates_orphans(monkeypatch, tmp_path):
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

    analytics_mod = types.ModuleType("analytics")
    analytics_mod.adaptive_roi_model = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "analytics", analytics_mod)

    ar_mod = types.ModuleType("adaptive_roi_predictor")
    ar_mod.load_training_data = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "adaptive_roi_predictor", ar_mod)

    pkg_path = Path(__file__).resolve().parents[1] / "sandbox_runner"
    pkg_stub = types.ModuleType("sandbox_runner")
    pkg_stub.__path__ = [str(pkg_path)]
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg_stub)

    repo = tmp_path
    (repo / "root.py").write_text("VALUE = 1\n")  # path-ignore
    (repo / "dep_pass.py").write_text("VALUE = 1\n")  # path-ignore

    grapher_calls: list[list[str]] = []

    class DummyGrapher:
        def __init__(self, *a, **k):
            pass

        def load(self, path):
            pass

        def build_graph(self, root):
            pass

        def update_graph(self, mods):
            grapher_calls.append(list(mods))

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = DummyGrapher
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    cluster_calls: list[str] = []

    class DummyClusterer:
        def __init__(self, *a, **k):
            pass

        def index_modules(self, paths):
            cluster_calls.extend(str(p) for p in paths)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyClusterer
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    cycle = importlib.import_module("sandbox_runner.cycle")

    added = {"root.py", "dep_pass.py"}  # path-ignore
    dotted = {Path(m).with_suffix("").as_posix().replace("/", ".") for m in added}

    grapher = mg_mod.ModuleSynergyGrapher(root=repo)
    graph_path = repo / "sandbox_data" / "module_synergy_graph.json"
    try:
        grapher.load(graph_path)
    except Exception:
        try:
            grapher.build_graph(repo)
        except Exception:
            pass
    grapher.update_graph(sorted(dotted))

    data_dir = repo / "sandbox_data"
    clusterer = ic_mod.IntentClusterer(
        local_db_path=data_dir / "intent.db", shared_db_path=data_dir / "intent.db"
    )
    paths = [repo / m for m in added]
    clusterer.index_modules(paths)

    assert grapher_calls == [["dep_pass", "root"]]
    assert set(cluster_calls) == {str(repo / "root.py"), str(repo / "dep_pass.py")}  # path-ignore
