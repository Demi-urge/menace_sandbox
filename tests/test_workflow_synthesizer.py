import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pytest
from analysis.io_signature import ModuleSignature

# Stub heavy optional dependencies before importing the module under test.
sys.modules.setdefault("intent_clusterer", SimpleNamespace(IntentClusterer=None))
sys.modules.setdefault(
    "module_synergy_grapher",
    SimpleNamespace(ModuleSynergyGrapher=None, get_synergy_cluster=None),
)

import workflow_synthesizer as ws  # noqa: E402


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


def _write_io_module(tmp_path: Path) -> None:
    """Create a module exercising IO and globals for analysis tests."""

    (tmp_path / "mod_io.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "CONFIG = 'config.json'\n"
        "GLOBAL_VAR = 0\n\n"
        "def worker(arg1: str) -> str:\n"
        "    global GLOBAL_VAR\n"
        "    data = open(os.path.join('data', 'input.txt')).read()\n"
        "    GLOBAL_VAR = arg1\n"
        "    with Path(f\"out_{'file'}.txt\").open('w') as fh:\n"
        "        fh.write(data)\n"
        "    token = os.environ['TOKEN']\n"
        "    return token\n"
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
    assert steps[-1]["inputs"] == ["missing"]
    assert grapher.loaded == tmp_path / "graph.json"


def test_module_io_extraction(tmp_path, monkeypatch):
    _write_io_module(tmp_path)
    monkeypatch.chdir(tmp_path)

    info = ws.inspect_module("mod_io")
    worker = info.functions["worker"]
    assert worker["args"] == ["arg1"]
    assert worker["returns"] == "str"
    assert info.globals >= {"CONFIG", "GLOBAL_VAR"}
    assert "data/input.txt" in info.files_read
    assert "out_file.txt" in info.files_written
    assert "TOKEN" in info.env_vars

    analyzer = ws.ModuleIOAnalyzer(cache_path=tmp_path / "cache.json")
    analyzed = analyzer.analyze("mod_io.py")
    assert "TOKEN" in analyzed["inputs"]


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

    assert [s["module"] for s in spec["steps"]] == ["mod_a", "mod_b", "mod_c"]
    required_step_keys = {"module", "inputs", "outputs", "files", "globals"}
    assert all(required_step_keys.issubset(s) for s in spec["steps"])

    spec_path = ws.save_workflow(workflows[0])
    saved = json.loads(spec_path.read_text())
    assert saved == spec

    generated = list(Path("sandbox_data/generated_workflows").glob("*.workflow.json"))
    assert generated
    saved_spec = json.loads(generated[0].read_text())
    assert "steps" in saved_spec


def test_to_json_yaml_helpers():
    workflow = [{"module": "m", "inputs": ["x"], "outputs": ["y"]}]
    data = json.loads(ws.to_json(workflow))
    assert data == {"steps": workflow}

    yaml = pytest.importorskip("yaml")
    again = yaml.safe_load(ws.to_yaml(workflow))
    assert again == data


def test_dependency_resolution(tmp_path, monkeypatch):
    """Modules are ordered by produced values and missing deps raise errors."""

    _write_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    synth = ws.WorkflowSynthesizer()
    modules = [ws.inspect_module(m) for m in ["mod_b", "mod_c", "mod_a"]]
    steps = synth.resolve_dependencies(modules)
    assert [s.module for s in steps] == ["mod_a", "mod_b", "mod_c"]

    bad = [ws.inspect_module(m) for m in ["mod_a", "mod_d"]]
    with pytest.raises(ValueError, match="mod_d"):
        synth.resolve_dependencies(bad)


def test_synthesise_workflow_wrapper(tmp_path, monkeypatch):
    """Public wrapper exposes synthesizer functionality."""

    _write_modules(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = ws.synthesise_workflow(start="mod_a")
    assert "steps" in result
    assert result["steps"][0]["module"] == "mod_a"


def test_expand_cluster_uses_both_sources(monkeypatch):
    """Synergy grapher and intent clusterer are consulted."""

    synergy = MagicMock()
    synergy.get_synergy_cluster.return_value = ["mod_b"]
    intent = MagicMock()
    intent.find_modules_related_to.return_value = [
        SimpleNamespace(path="mod_c.py")
    ]

    synth = ws.WorkflowSynthesizer(
        module_synergy_grapher=synergy, intent_clusterer=intent
    )
    modules = synth.expand_cluster(start_module="mod_a", problem="x")

    synergy.get_synergy_cluster.assert_called_once()
    intent.find_modules_related_to.assert_called_once()
    assert modules == {"mod_a", "mod_b", "mod_c"}


def test_resolve_dependencies_missing_inputs():
    """Missing producers raise a ``ValueError``."""

    class DummyGrapher:
        graph = None

        def load(self, path):  # pragma: no cover - trivial
            pass

    synth = ws.WorkflowSynthesizer(module_synergy_grapher=DummyGrapher())

    a = ModuleSignature(name="a")
    a.files_written.add("data.txt")
    a.functions = {"f": {"args": [], "annotations": {}, "returns": None}}

    b = ModuleSignature(name="b")
    b.files_read.add("data.txt")
    b.functions = {"f": {"args": [], "annotations": {}, "returns": None}}

    c = ModuleSignature(name="c")
    c.files_read.add("missing.txt")
    c.functions = {"f": {"args": [], "annotations": {}, "returns": None}}

    with pytest.raises(ValueError) as exc:
        synth.resolve_dependencies([a, b, c])
    assert "Unresolved dependencies" in str(exc.value)
    assert "c" in str(exc.value)


def test_resolve_dependencies_cycle_detection():
    """Dependency cycles are reported via ``ValueError``."""

    class DummyGrapher:
        graph = None

        def load(self, path):  # pragma: no cover - trivial
            pass

    synth = ws.WorkflowSynthesizer(module_synergy_grapher=DummyGrapher())

    a = ModuleSignature(name="a")
    a.globals.add("a_out")
    a.functions = {
        "fa": {"args": ["b_out"], "annotations": {}, "returns": None}
    }

    b = ModuleSignature(name="b")
    b.globals.add("b_out")
    b.functions = {
        "fb": {"args": ["a_out"], "annotations": {}, "returns": None}
    }

    with pytest.raises(ValueError) as exc:
        synth.resolve_dependencies([a, b])
    assert "Cyclic dependency detected" in str(exc.value)


def test_cli_end_to_end(tmp_path):
    """CLI produces a JSON workflow when executed."""

    stub = tmp_path / "stubs"
    stub.mkdir()
    (stub / "module_synergy_grapher.py").write_text(
        "class ModuleSynergyGrapher:\n"
        "    def __init__(self):\n"
        "        self.graph=None\n"
        "    def load(self, path):\n"
        "        pass\n"
        "    def get_synergy_cluster(self, start_module, threshold=0.0):\n"
        "        return []\n"
        "\n"
        "def get_synergy_cluster(start_module, path=None, threshold=0.0):\n"
        "    return []\n"
        "\n"
        "def load_graph(path):\n"
        "    return None\n"
    )
    (stub / "intent_clusterer.py").write_text(
        "class IntentClusterer:\n"
        "    def find_modules_related_to(self, problem, top_k=20):\n"
        "        return []\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(stub), os.getcwd()])
    result = subprocess.run(
        [sys.executable, "workflow_synthesizer_cli.py", "simple_functions"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    data = json.loads(result.stdout)
    assert data
    assert data[0]["steps"][0]["module"] == "simple_functions"


def test_cli_save_and_list(tmp_path):
    """`--save` writes workflow files and `--list` reports them."""

    stub = tmp_path / "stubs"
    stub.mkdir()
    (stub / "module_synergy_grapher.py").write_text(
        "class ModuleSynergyGrapher:\n"
        "    def __init__(self):\n"
        "        self.graph=None\n"
        "    def load(self, path):\n"
        "        pass\n"
        "    def get_synergy_cluster(self, start_module, threshold=0.0):\n"
        "        return []\n"
        "\n"
        "def get_synergy_cluster(start_module, path=None, threshold=0.0):\n"
        "    return []\n"
        "\n"
        "def load_graph(path):\n"
        "    return None\n"
    )
    (stub / "intent_clusterer.py").write_text(
        "class IntentClusterer:\n"
        "    def find_modules_related_to(self, problem, top_k=20):\n"
        "        return []\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(stub), os.getcwd()])
    cli = Path(__file__).resolve().parent.parent / "workflow_synthesizer_cli.py"

    subprocess.run(
        [sys.executable, str(cli), "simple_functions", "--save"],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    saved = tmp_path / "sandbox_data" / "generated_workflows" / "simple_functions.workflow.json"
    assert saved.is_file()

    result = subprocess.run(
        [sys.executable, str(cli), "--list"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=tmp_path,
    )
    assert "simple_functions.workflow.json" in result.stdout
