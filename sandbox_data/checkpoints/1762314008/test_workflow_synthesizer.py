import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from dynamic_path_router import resolve_path

import networkx as nx
import pytest

# Stub optional heavy dependencies before importing module under test.
sys.modules["intent_clusterer"] = SimpleNamespace(IntentClusterer=None)
sys.modules[
    "module_synergy_grapher"
] = SimpleNamespace(ModuleSynergyGrapher=None, get_synergy_cluster=None)
sys.modules["intent_db"] = SimpleNamespace(IntentDB=None)

import workflow_synthesizer as ws  # noqa: E402


def _copy_modules(tmp_path: Path) -> None:
    for mod in resolve_path("tests/fixtures/workflow_modules").glob("*.py"):  # path-ignore
        shutil.copy(mod, tmp_path / mod.name)


class StubGrapher:
    """Tiny synergy grapher with a weighted graph."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edge("mod_a", "mod_b", weight=2.0)
        self.graph.add_edge("mod_b", "mod_c", weight=1.0)
        self.graph.add_edge("mod_c", "mod_d", weight=1.0)
        # edges for multi-return fixtures
        self.graph.add_edge("mod_i", "mod_j", weight=1.0)
        self.graph.add_edge("mod_k", "mod_l", weight=1.0)
        self.graph.add_edge("mod_m", "mod_n", weight=1.0)


class StubIntent:
    """Return ``mod_c`` as intent match."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.top_k_seen: int | None = None

    def find_modules_related_to(self, _problem: str, top_k: int = 10):  # pragma: no cover - simple
        self.top_k_seen = top_k
        return [SimpleNamespace(path=str(self.base / "mod_c.py"), score=1.0)]  # path-ignore


def test_expand_cluster_merges_sources(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    intent = StubIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher, intent_clusterer=intent)

    modules = synth.expand_cluster(
        start_module="mod_a", problem="finalise", max_depth=2
    )
    assert modules == {"mod_a", "mod_b", "mod_c"}


def test_expand_cluster_bfs_multi_hop(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)

    # depth 1 restricts exploration to the direct neighbour
    assert synth.expand_cluster(start_module="mod_a", max_depth=1) == {
        "mod_a",
        "mod_b",
    }

    # expanding to depth 2 pulls in mod_c
    modules = synth.expand_cluster(start_module="mod_a", max_depth=2)
    assert modules == {"mod_a", "mod_b", "mod_c"}

    # depth 3 includes mod_d; threshold filters out weaker edges
    modules = synth.expand_cluster(start_module="mod_a", max_depth=3)
    assert modules == {"mod_a", "mod_b", "mod_c", "mod_d"}

    modules = synth.expand_cluster(start_module="mod_a", max_depth=3, threshold=1.5)
    assert modules == {"mod_a", "mod_b"}


def test_expand_cluster_intent_limit(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    intent = StubIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(intent_clusterer=intent)

    synth.expand_cluster(problem="finalise", intent_limit=5)
    assert intent.top_k_seen == 5


def test_resolve_dependencies_ordering_and_unresolved(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    synth = ws.WorkflowSynthesizer()
    mods = [ws.inspect_module(m) for m in ["mod_b", "mod_c", "mod_a"]]
    steps = synth.resolve_dependencies(mods)
    order = [s.module for s in steps]
    assert order[0] == "mod_a"
    assert sorted(order[1:]) == ["mod_b", "mod_c"]

    bad = [ws.inspect_module(m) for m in ["mod_a", "mod_d"]]
    steps = synth.resolve_dependencies(bad)
    unresolved = {s.module: s.unresolved for s in steps}
    assert unresolved["mod_d"] == ["missing"]


def test_resolve_dependencies_types_and_files(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    synth = ws.WorkflowSynthesizer()
    mods = [ws.inspect_module(m) for m in ["mod_f", "mod_h", "mod_e", "mod_g"]]
    steps = synth.resolve_dependencies(mods)
    order = [s.module for s in steps]
    assert order.index("mod_e") < order.index("mod_f")
    assert order.index("mod_g") < order.index("mod_h")


def test_resolve_dependencies_rich_returns(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    synth = ws.WorkflowSynthesizer()
    mods = [
        ws.inspect_module(m)
        for m in ["mod_j", "mod_l", "mod_n", "mod_i", "mod_k", "mod_m"]
    ]
    steps = synth.resolve_dependencies(mods)
    order = [s.module for s in steps]
    assert order.index("mod_i") < order.index("mod_j")
    assert order.index("mod_k") < order.index("mod_l")
    assert order.index("mod_m") < order.index("mod_n")


def test_resolve_dependencies_optional_args(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    synth = ws.WorkflowSynthesizer()
    mods = [ws.inspect_module(m) for m in ["mod_o", "mod_p"]]
    steps = synth.resolve_dependencies(mods)
    order = [s.module for s in steps]
    assert order.index("mod_o") < order.index("mod_p")
    unresolved = {s.module: s.unresolved for s in steps}
    assert unresolved["mod_p"] == []


def test_resolve_dependencies_optional_args_missing(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    synth = ws.WorkflowSynthesizer()
    mods = [ws.inspect_module(m) for m in ["mod_p"]]
    steps = synth.resolve_dependencies(mods)
    unresolved = {s.module: s.unresolved for s in steps}
    assert unresolved["mod_p"] == ["data"]


def test_generate_workflows_persist_and_rank(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    intent = StubIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher, intent_clusterer=intent)

    workflows = synth.generate_workflows(
        start_module="mod_a", problem="finalise", limit=2, max_depth=2
    )
    assert len(workflows) == 2
    assert [step.module for step in workflows[0]] == ["mod_a", "mod_b"]
    # The second best workflow follows the next step in the synergy chain
    assert [step.module for step in workflows[1]] == ["mod_a", "mod_b", "mod_c"]

    out_dir = resolve_path("sandbox_data") / "generated_workflows"
    saved = sorted(out_dir.glob("*.workflow.json"))
    assert len(saved) >= 2 and saved[0].name.startswith("mod_a_0")

    data = json.loads(saved[0].read_text())
    assert data["steps"][0]["module"] == "mod_a"
    # Scores are stored in descending order
    assert synth.workflow_scores == sorted(synth.workflow_scores, reverse=True)
    assert synth.workflow_score_details[1]["intent"] == pytest.approx(1 / 3)
    assert 0.0 <= synth.workflow_score_details[1]["intent"] <= 1.0


def test_generate_workflows_auto_evaluate(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)

    results = [True, False]

    def fake_eval(_spec):
        return results.pop(0)

    monkeypatch.setattr(ws, "evaluate_workflow", fake_eval)

    synth.generate_workflows(
        start_module="mod_a", limit=2, max_depth=2, auto_evaluate=True
    )
    assert synth.workflow_score_details[0]["success"] is True
    assert synth.workflow_score_details[1]["success"] is False

    saved = sorted(
        (resolve_path("sandbox_data") / "generated_workflows").glob("*.workflow.json")
    )
    data = json.loads(saved[0].read_text())
    assert data.get("success") is True


def test_generate_workflows_max_depth(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)

    workflows = synth.generate_workflows(start_module="mod_a", limit=5, max_depth=1)
    flat = [[step.module for step in wf] for wf in workflows]
    assert all("mod_c" not in order for order in flat)

    workflows = synth.generate_workflows(start_module="mod_a", limit=5, max_depth=2)
    flat = [[step.module for step in wf] for wf in workflows]
    assert any(order == ["mod_a", "mod_b", "mod_c"] for order in flat)


def test_generate_workflows_min_score_prunes(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)

    workflows = synth.generate_workflows(
        start_module="mod_a", limit=5, max_depth=2, min_score=0.1
    )
    assert [[step.module for step in wf] for wf in workflows] == [["mod_a"]]


def test_generate_workflows_rich_returns(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)

    workflows = synth.generate_workflows(start_module="mod_i", limit=1, max_depth=1)
    assert [step.module for step in workflows[0]] == ["mod_i", "mod_j"]

    workflows = synth.generate_workflows(start_module="mod_k", limit=1, max_depth=1)
    assert [step.module for step in workflows[0]] == ["mod_k", "mod_l"]

    workflows = synth.generate_workflows(start_module="mod_m", limit=1, max_depth=1)
    assert [step.module for step in workflows[0]] == ["mod_m", "mod_n"]


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


def test_generate_workflows_penalties_and_tiebreak(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)

    workflows = synth.generate_workflows(
        start_module="mod_a",
        limit=5,
        max_depth=3,
        synergy_weight=0.0,
        intent_weight=0.0,
    )
    flat = [[step.module for step in wf] for wf in workflows]

    assert flat[0] == ["mod_a"]
    assert flat[1] == ["mod_a", "mod_b"]
    assert synth.workflow_scores[0] == synth.workflow_scores[1]

    # The deepest explored path incurs the highest penalty
    assert flat[-1] == ["mod_a", "mod_b", "mod_c", "mod_d"]
    assert synth.workflow_scores[-1] < 0


class StubIntentDB:
    def __init__(self, base: Path) -> None:
        self.base = base

    def encode_text(self, _text: str):  # pragma: no cover - simple
        return [0.1]

    def search_by_vector(self, _vec, top_k: int = 50):  # pragma: no cover - simple
        return [(str(self.base / "mod_b.py"), 0.1)]  # path-ignore


def test_generate_workflows_intent_db_scoring(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher)
    synth.intent_db = StubIntentDB(tmp_path)
    synth.intent_clusterer = None

    workflows = synth.generate_workflows(
        start_module="mod_a", problem="finalise", limit=2, max_depth=2
    )

    assert len(workflows) == 2
    assert [step.module for step in workflows[0]] == ["mod_a", "mod_b"]
    assert [step.module for step in workflows[1]] == ["mod_a", "mod_b", "mod_c"]
    assert synth.workflow_score_details[0]["intent"] == pytest.approx(0.5)
    assert synth.workflow_score_details[1]["intent"] == pytest.approx(1 / 3)


def test_scoring_normalisation_and_penalty(tmp_path, monkeypatch):
    _copy_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = StubGrapher()

    class MultiIntent:
        def __init__(self, base: Path) -> None:
            self.base = base

        def find_modules_related_to(self, _problem: str, top_k: int = 10):
            return [
                SimpleNamespace(path=str(self.base / "mod_b.py"), score=0.5),  # path-ignore
                SimpleNamespace(path=str(self.base / "mod_c.py"), score=1.0),  # path-ignore
            ]

    intent = MultiIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(module_synergy_grapher=grapher, intent_clusterer=intent)
    workflows = synth.generate_workflows(
        start_module="mod_a",
        problem="finalise",
        limit=5,
        max_depth=3,
        synergy_weight=0.6,
        intent_weight=0.4,
    )

    first = synth.workflow_score_details[0]
    assert [step.module for step in workflows[0]] == ["mod_a", "mod_b"]
    assert first["synergy"] == pytest.approx(1.0)
    assert first["intent"] == pytest.approx(0.5 / 2)
    expected = 0.6 * first["synergy"] + 0.4 * first["intent"] - first["penalty"]
    assert synth.workflow_scores[0] == pytest.approx(expected)

    for wf, info, score in zip(
        synth.generated_workflows, synth.workflow_score_details, synth.workflow_scores
    ):
        modules = [step.module for step in wf]
        if modules == ["mod_a", "mod_b", "mod_c"]:
            assert info["penalty"] == 1
            expected = 0.6 * info["synergy"] + 0.4 * info["intent"] - info["penalty"]
            assert score == pytest.approx(expected)
            break
    else:  # pragma: no cover - ensure penalty path exists
        pytest.fail("penalised workflow missing")
