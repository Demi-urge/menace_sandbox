import shutil
from pathlib import Path

import networkx as nx

import workflow_synthesizer as ws
from dynamic_path_router import resolve_path


def _copy_modules(tmp_path: Path) -> None:
    for mod in resolve_path("tests/fixtures/workflow_modules").glob("*.py"):  # path-ignore
        shutil.copy(mod, tmp_path / mod.name)


class DummyGrapher:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edge("mod_a", "mod_b", weight=1.0)
        self.graph.add_edge("mod_b", "mod_c", weight=1.0)


def test_expand_cluster_basic(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=DummyGrapher())
    mods = synth.expand_cluster(start_module="mod_a", max_depth=2)
    assert mods == {"mod_a", "mod_b", "mod_c"}
    mods = synth.expand_cluster(start_module="mod_a", max_depth=2, threshold=1.5)
    assert mods == {"mod_a"}


def test_resolve_dependencies_basic(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)
    synth = ws.WorkflowSynthesizer()
    mods = [ws.inspect_module(m) for m in ["mod_b", "mod_a"]]
    steps = synth.resolve_dependencies(mods)
    assert [s.module for s in steps] == ["mod_a", "mod_b"]
    bad = [ws.inspect_module("mod_d")]
    steps = synth.resolve_dependencies(bad)
    assert steps[0].unresolved == ["missing"]


def test_generate_workflows_limit(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=DummyGrapher())
    workflows = synth.generate_workflows(start_module="mod_a", limit=1, max_depth=2)
    assert len(workflows) == 1
    assert [s.module for s in workflows[0]] == ["mod_a", "mod_b"]


def test_synthesize_routing(tmp_path, monkeypatch):
    (tmp_path / "mod_a.py").write_text("def start():\n    return 1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)

    called = {}

    def fake_greedy(self, start_module=None, problem=None, threshold=0.0, **kwargs):
        called["start_module"] = start_module
        called["problem"] = problem
        called["threshold"] = threshold
        return [{"module": start_module or "mod_z", "inputs": [], "outputs": []}]

    monkeypatch.setattr(ws.WorkflowSynthesizer, "_synthesize_greedy", fake_greedy)
    synth = ws.WorkflowSynthesizer()

    result = synth.synthesize("mod_a", threshold=0.2)
    assert called == {"start_module": "mod_a", "problem": None, "threshold": 0.2}
    assert result["steps"][0]["module"] == "mod_a"

    result = synth.synthesize("a problem")
    assert called["start_module"] is None
    assert called["problem"] == "a problem"
    assert result["steps"][0]["module"] == "mod_z"
