import ast
from pathlib import Path
from dynamic_path_router import resolve_path
import os
from types import SimpleNamespace


def test_evolve_workflows_calls_evolver():
    class WorkflowDB:
        def __init__(self, *a, **k):
            pass

        def fetch_workflows(self, limit=10):
            return [{"workflow": ["a", "b"], "id": 1}]

    class WorkflowGraph:
        def __init__(self, *a, **k):
            self.graph = {"nodes": {}}

    class Evolver:
        def __init__(self):
            self.called = []

        def build_callable(self, seq):
            return lambda: True

        def evolve(self, fn, wf_id, variants=5):
            self.called.append(wf_id)
            out = lambda: False  # new callable so promoted
            out.parent_id = wf_id
            out.mutation_description = "variant"
            return out

        def is_stable(self, wf_id):
            return False

    ns = {
        "WorkflowDB": WorkflowDB,
        "WorkflowRecord": object,
        "WorkflowGraph": WorkflowGraph,
        "log_record": lambda **k: k,
        "Path": Path,
        "os": os,
    }

    src = resolve_path("self_improvement.py").read_text()
    tree = ast.parse(src)
    class_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
    )
    method_node = next(
        m for m in class_node.body if isinstance(m, ast.FunctionDef) and m.name == "_evolve_workflows"
    )
    module = ast.Module([method_node], type_ignores=[])
    exec(compile(module, "<ast>", "exec"), ns)
    evolve = ns["_evolve_workflows"]

    ev = Evolver()
    self_obj = SimpleNamespace(
        workflow_evolver=ev,
        pathway_db=None,
        logger=SimpleNamespace(exception=lambda *a, **k: None),
        meta_logger=None,
    )
    results = evolve(self_obj)
    assert ev.called == [1]
    assert results[1]["status"] == "promoted"
    assert results[1]["parent_id"] == 1
    assert results[1]["mutation_description"] == "variant"


def test_evolve_workflows_skips_flagged_workflow():
    class WorkflowDB:
        def __init__(self, *a, **k):
            pass

        def fetch_workflows(self, limit=10):
            return [{"workflow": ["a", "b"], "id": 1}]

    class WorkflowGraph:
        def __init__(self, *a, **k):
            self.graph = {"nodes": {}}

    class Evolver:
        def build_callable(self, seq):
            return lambda: True

        def evolve(self, fn, wf_id, variants=5):
            return fn

        def is_stable(self, wf_id):
            return False

    class MetaLogger:
        def __init__(self):
            self.module_deltas = {"workflow:1": [0.0, 0.0, 0.0]}
            self.flagged_sections = set()

        def diminishing(self, threshold):
            return ["workflow:1"]

    ns = {
        "WorkflowDB": WorkflowDB,
        "WorkflowRecord": object,
        "WorkflowGraph": WorkflowGraph,
        "log_record": lambda **k: k,
        "Path": Path,
        "os": os,
    }

    src = resolve_path("self_improvement.py").read_text()
    tree = ast.parse(src)
    class_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
    )
    method_node = next(
        m for m in class_node.body if isinstance(m, ast.FunctionDef) and m.name == "_evolve_workflows"
    )
    module = ast.Module([method_node], type_ignores=[])
    exec(compile(module, "<ast>", "exec"), ns)
    evolve = ns["_evolve_workflows"]

    self_obj = SimpleNamespace(
        workflow_evolver=Evolver(),
        pathway_db=None,
        logger=SimpleNamespace(exception=lambda *a, **k: None),
        meta_logger=MetaLogger(),
    )
    results = evolve(self_obj)
    assert results == {}
