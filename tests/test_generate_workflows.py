import importlib.util
import sys
import types
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]


def _load_env():
    pkg = sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
    pkg.__path__ = [str(ROOT / "sandbox_runner")]
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.environment", ROOT / "sandbox_runner" / "environment.py"  # path-ignore
    )
    env = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner.environment"] = env
    assert spec and spec.loader
    spec.loader.exec_module(env)  # type: ignore[attr-defined]
    return env


def test_generate_workflows(tmp_path):
    # stub import graph to avoid scanning repository
    dmm = types.ModuleType("dynamic_module_mapper")
    dmm.build_import_graph = lambda root: nx.DiGraph()
    sys.modules["dynamic_module_mapper"] = dmm

    vs = types.ModuleType("vector_service")
    vs.EmbeddableDBMixin = type("EmbeddableDBMixin", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["vector_service"] = vs

    thb = types.ModuleType("menace.task_handoff_bot")
    _db: dict[str, list] = {}

    class WorkflowRecord:
        def __init__(self, workflow, title="", dependencies=None, reasons=None):
            self.workflow = workflow
            self.title = title
            self.dependencies = dependencies or []
            self.reasons = reasons or []
            self.wid = 0

    class WorkflowDB:
        def __init__(self, path, router=None):
            self.path = str(Path(path))
            self.records = _db.setdefault(self.path, [])

        def add(self, rec, source_menace_id=""):
            rec.wid = len(self.records) + 1
            self.records.append(rec)
            return rec.wid

        def fetch(self):
            return list(self.records)

    thb.WorkflowDB = WorkflowDB
    thb.WorkflowRecord = WorkflowRecord
    sys.modules["menace.task_handoff_bot"] = thb

    env = _load_env()
    db_path = tmp_path / "wf.db"
    # avoid expensive orphan integration
    env.integrate_new_orphans = lambda repo, router=None: []
    ids = env.generate_workflows_for_modules(["foo.py"], workflows_db=db_path)  # path-ignore
    recs = thb.WorkflowDB(db_path).fetch()
    assert ids and ids[0] == recs[0].wid
    assert recs[0].workflow == ["foo"]


def test_generate_workflows_with_dependencies(tmp_path, monkeypatch):
    g = nx.DiGraph()
    g.add_edge("b", "a")
    dmm = types.ModuleType("dynamic_module_mapper")
    dmm.build_import_graph = lambda root: g
    sys.modules["dynamic_module_mapper"] = dmm

    vs = types.ModuleType("vector_service")
    vs.EmbeddableDBMixin = type("EmbeddableDBMixin", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["vector_service"] = vs

    thb = types.ModuleType("menace.task_handoff_bot")
    _db: dict[str, list] = {}

    class WorkflowRecord:
        def __init__(self, workflow, title="", dependencies=None, reasons=None):
            self.workflow = workflow
            self.title = title
            self.dependencies = dependencies or []
            self.reasons = reasons or []
            self.wid = 0

    class WorkflowDB:
        def __init__(self, path, router=None):
            self.path = str(Path(path))
            self.records = _db.setdefault(self.path, [])

        def add(self, rec, source_menace_id=""):
            rec.wid = len(self.records) + 1
            self.records.append(rec)
            return rec.wid

        def fetch(self):
            return list(self.records)

    thb.WorkflowDB = WorkflowDB
    thb.WorkflowRecord = WorkflowRecord
    sys.modules["menace.task_handoff_bot"] = thb

    env = _load_env()
    db_path = tmp_path / "wf.db"
    monkeypatch.setattr(env, "integrate_new_orphans", lambda repo, router=None: [])
    ids = env.generate_workflows_for_modules(["b.py"], workflows_db=db_path)  # path-ignore
    rec = thb.WorkflowDB(db_path).fetch()[0]
    assert rec.workflow == ["a", "b"]
    assert rec.dependencies == ["a"]
    assert any("b" in r and "a" in r for r in rec.reasons)
