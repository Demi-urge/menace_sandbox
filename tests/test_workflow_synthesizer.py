import json
import sys
from pathlib import Path
from types import SimpleNamespace
import networkx as nx

import workflow_synthesizer as ws
import workflow_spec as wspec


def _write_modules(tmp_path):
    (tmp_path / "mod_a.py").write_text(
        "def start():\n    data = 'x'\n    return data\n"
    )
    (tmp_path / "mod_b.py").write_text(
        "def middle(data):\n    result = data + 'b'\n    return result\n"
    )
    (tmp_path / "mod_c.py").write_text(
        "def end(result):\n    pass\n"
    )


class FakeGrapher:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_edge("mod_a", "mod_b", weight=2.0)
        self.graph.add_edge("mod_b", "mod_c", weight=1.0)
        self.loaded = None

    def load(self, path):
        self.loaded = path

    def get_synergy_cluster(self, start):
        return ["mod_a", "mod_b"] if start == "mod_a" else [start]


class FakeIntent:
    def __init__(self, base):
        self.base = base

    def find_modules_related_to(self, _problem, top_k=10):
        return [SimpleNamespace(path=str(self.base / "mod_c.py"), score=1.0)]


def test_workflow_synthesizer_roundtrip(tmp_path, monkeypatch):
    _write_modules(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    import menace.task_handoff_bot as thb  # preload with package context
    sys.modules["task_handoff_bot"] = thb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKFLOW_OUTPUT_DIR", str(tmp_path))

    grapher = FakeGrapher()
    intent = FakeIntent(tmp_path)
    synth = ws.WorkflowSynthesizer(
        module_synergy_grapher=grapher,
        intent_clusterer=intent,
        synergy_graph_path=tmp_path / "graph.json",
    )

    modules = synth.synthesize(start_module="mod_a", problem="finish")
    assert set(modules) == {"mod_a", "mod_b", "mod_c"}
    assert grapher.loaded == tmp_path / "graph.json"

    workflows = synth.generate_workflows("mod_a", problem="finish")
    assert [step["module"] for step in workflows[0]] == ["mod_a", "mod_b", "mod_c"]

    steps = [
        {"name": s["module"], "bot": s["module"], "args": s["inputs"]}
        for s in workflows[0]
    ]
    spec = wspec.to_spec(steps)
    path = wspec.save(spec)
    data = json.loads(path.read_text())
    assert data["workflow"] == ["mod_a", "mod_b", "mod_c"]

    WorkflowDB, _ = wspec._load_thb()
    db = WorkflowDB(tmp_path / "workflows.db")
    recs = db.fetch()
    assert recs and recs[0].workflow == ["mod_a", "mod_b", "mod_c"]
