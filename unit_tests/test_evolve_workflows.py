from types import SimpleNamespace
import ast
from pathlib import Path
import os


def test_evolve_workflows_benchmarks_variants():
    class WorkflowDB:
        def __init__(self, *a, **k):
            pass

        def fetch_workflows(self, limit=10):
            return [{"workflow": ["a", "b"], "id": 1}]

    class WorkflowGraph:
        def __init__(self, *a, **k):
            self.graph = {"nodes": {}}

    class WorkflowEvolutionBot:
        def __init__(self, *a, **k):
            pass

        def generate_variants(self, workflow_id):
            yield "b-a"

    class CompositeWorkflowScorer:
        def __init__(self, *a, **k):
            pass

        def run(self, fn, wf_id, run_id):
            roi = 1.0 if run_id == "baseline" else 2.0
            return SimpleNamespace(roi_gain=roi, runtime=0.0, success_rate=1.0)

    class ROIResultsDB:
        def __init__(self, *a, **k):
            pass

    ns = {
        "WorkflowDB": WorkflowDB,
        "WorkflowRecord": object,
        "WorkflowGraph": WorkflowGraph,
        "WorkflowEvolutionBot": WorkflowEvolutionBot,
        "CompositeWorkflowScorer": CompositeWorkflowScorer,
        "ROIResultsDB": ROIResultsDB,
        "EvaluationResult": SimpleNamespace,
        "log_record": lambda **k: k,
        "Path": Path,
        "os": os,
    }

    src = Path("self_improvement_engine.py").read_text()
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
        workflow_evolver=SimpleNamespace(build_callable=lambda seq: lambda: True),
        pathway_db=None,
        roi_tracker=None,
        logger=SimpleNamespace(exception=lambda *a, **k: None),
    )
    results = evolve(self_obj)
    assert results[1]["baseline"] == 1.0
    assert results[1]["best"] == 2.0
    assert results[1]["sequence"] == "b-a"
