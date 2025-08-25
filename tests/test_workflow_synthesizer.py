import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

# Stub optional heavy dependencies before importing module under test.
sys.modules.setdefault("intent_clusterer", SimpleNamespace(IntentClusterer=None))
sys.modules.setdefault(
    "module_synergy_grapher",
    SimpleNamespace(ModuleSynergyGrapher=None, get_synergy_cluster=None),
)

import workflow_synthesizer as ws  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "workflow_modules"


def _copy_modules(tmp_path: Path) -> None:
    for mod in FIXTURES.glob("*.py"):
        shutil.copy(mod, tmp_path / mod.name)


class StubGrapher:
    """Tiny synergy grapher with a weighted graph."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edge("mod_a", "mod_b", weight=2.0)
        self.graph.add_edge("mod_b", "mod_c", weight=1.0)
        self.graph.add_edge("mod_c", "mod_d", weight=1.0)


class StubIntent:
    """Return ``mod_c`` as intent match."""

    def __init__(self, base: Path) -> None:
        self.base = base

    def find_modules_related_to(self, _problem: str, top_k: int = 10):  # pragma: no cover - simple
        return [SimpleNamespace(path=str(self.base / "mod_c.py"), score=1.0)]


def test_expand_cluster_merges_sources(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    intent = StubIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher, intent_clusterer=intent)

    modules = synth.expand_cluster(start_module="mod_a", problem="finalise")
    assert modules == {"mod_a", "mod_b", "mod_c"}


def test_expand_cluster_bfs_multi_hop(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)

    # default depth is 1 -> only direct neighbour
    assert synth.expand_cluster(start_module="mod_a") == {"mod_a", "mod_b"}

    # expanding to depth 2 pulls in mod_c
    modules = synth.expand_cluster(start_module="mod_a", max_depth=2)
    assert modules == {"mod_a", "mod_b", "mod_c"}

    # depth 3 includes mod_d; threshold filters out weaker edges
    modules = synth.expand_cluster(start_module="mod_a", max_depth=3)
    assert modules == {"mod_a", "mod_b", "mod_c", "mod_d"}

    modules = synth.expand_cluster(start_module="mod_a", max_depth=3, threshold=1.5)
    assert modules == {"mod_a", "mod_b"}


def test_resolve_dependencies_ordering_and_unresolved(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    synth = ws.WorkflowSynthesizer()
    mods = [ws.inspect_module(m) for m in ["mod_b", "mod_c", "mod_a"]]
    steps = synth.resolve_dependencies(mods)
    assert [s.module for s in steps] == ["mod_a", "mod_b", "mod_c"]

    bad = [ws.inspect_module(m) for m in ["mod_a", "mod_d"]]
    with pytest.raises(ValueError, match="mod_d"):
        synth.resolve_dependencies(bad)


def test_generate_workflows_persist_and_rank(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    intent = StubIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher, intent_clusterer=intent)

    workflows = synth.generate_workflows(start_module="mod_a", problem="finalise")
    assert [step.module for step in workflows[0]] == ["mod_a", "mod_b", "mod_c"]
    out_dir = Path("sandbox_data/generated_workflows")
    saved = list(out_dir.glob("*.workflow.json"))
    assert saved and saved[0].name.startswith("mod_a_0")

    data = json.loads(saved[0].read_text())
    assert data["steps"][0]["module"] == "mod_a"
    # Scores are stored in descending order
    assert synth.workflow_scores == sorted(synth.workflow_scores, reverse=True)


def test_resolve_dependencies_cycles():
    synth = ws.WorkflowSynthesizer()

    a = ws.ModuleSignature(name="a")
    a.globals.add("a_out")
    a.functions = {"fa": {"args": ["b_out"], "annotations": {}, "returns": None}}

    b = ws.ModuleSignature(name="b")
    b.globals.add("b_out")
    b.functions = {"fb": {"args": ["a_out"], "annotations": {}, "returns": None}}

    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        synth.resolve_dependencies([a, b])
