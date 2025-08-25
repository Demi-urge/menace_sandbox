import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

# Stub heavy optional dependencies before importing the module under test.
sys.modules.setdefault("intent_clusterer", SimpleNamespace(IntentClusterer=None))
sys.modules.setdefault(
    "module_synergy_grapher",
    SimpleNamespace(ModuleSynergyGrapher=None, get_synergy_cluster=None),
)

import workflow_synthesizer as ws


def _write_modules(tmp_path: Path) -> None:
    """Create tiny modules for synthesizer tests."""

    (tmp_path / "mod_a.py").write_text(
        "def start():\n    data = 'a'\n    return data\n"
    )
    (tmp_path / "mod_b.py").write_text(
        "def middle(data):\n    result = data + 'b'\n    return result\n"
    )
    (tmp_path / "mod_c.py").write_text(
        "def final(result):\n    pass\n"
    )
    (tmp_path / "mod_d.py").write_text(
        "def unrelated(missing):\n    pass\n"
    )


class FakeGrapher:
    """Minimal synergy grapher used to control expansion."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edge("mod_a", "mod_b", weight=2.0)
        self.graph.add_edge("mod_a", "mod_d", weight=1.0)
        self.graph.add_edge("mod_b", "mod_c", weight=1.0)
        self.loaded: Path | None = None

    def load(self, path: Path) -> None:
        self.loaded = path

    def get_synergy_cluster(self, start: str):  # pragma: no cover - simple
        return ["mod_a", "mod_b", "mod_d"] if start == "mod_a" else [start]


class FakeIntent:
    """Return ``mod_c`` as an intent match."""

    def __init__(self, base: Path) -> None:
        self.base = base

    def find_modules_related_to(self, _problem: str, top_k: int = 10):  # pragma: no cover - trivial
        return [SimpleNamespace(path=str(self.base / "mod_c.py"), score=1.0)]


def test_cluster_expansion_and_io_matching(tmp_path, monkeypatch):
    _write_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    grapher = FakeGrapher()
    intent = FakeIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(
        module_synergy_grapher=grapher,
        intent_clusterer=intent,
        synergy_graph_path=tmp_path / "graph.json",
    )

    result = synth.synthesize({"module": "mod_a", "problem": "finalise"})
    steps = result["steps"]

    assert [s["module"] for s in steps] == ["mod_a", "mod_b", "mod_c", "mod_d"]
    assert steps[-1]["args"] == ["missing"]
    assert grapher.loaded == tmp_path / "graph.json"


def test_generated_json_schema(tmp_path, monkeypatch):
    _write_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    # Stub out WorkflowDB/Record used by workflow_spec.save
    @dataclass
    class DummyRecord:
        pass

    class DummyDB:
        def __init__(self, path):
            self.path = path
            self.records = []

        def add(self, rec):  # pragma: no cover - simple
            self.records.append(rec)

    sys.modules["task_handoff_bot"] = SimpleNamespace(
        WorkflowDB=DummyDB, WorkflowRecord=DummyRecord
    )

    grapher = FakeGrapher()
    intent = FakeIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(
        module_synergy_grapher=grapher, intent_clusterer=intent
    )

    workflows = synth.generate_workflows(start_module="mod_a", problem="finalise")
    spec = ws.to_workflow_spec(workflows[0])

    assert spec["workflow"] == ["mod_a", "mod_b", "mod_c"]
    required = {
        "workflow",
        "action_chains",
        "argument_strings",
        "assigned_bots",
        "enhancements",
        "title",
    }
    assert required.issubset(spec)

    spec_path = ws.save_workflow(workflows[0])
    saved = json.loads(spec_path.read_text())
    assert saved["workflow"] == spec["workflow"]

