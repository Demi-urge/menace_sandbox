import ast
import importlib
import logging
import types
from typing import Callable
from dynamic_path_router import resolve_path


def _load_build_callable():
    src = resolve_path("workflow_evolution_manager.py").read_text()
    tree = ast.parse(src)
    func_node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_callable"
    )
    module = ast.Module([func_node], type_ignores=[])
    env = {
        "importlib": importlib,
        "logger": logging.getLogger("test"),
        "Callable": Callable,
    }
    exec(compile(module, "<ast>", "exec"), env)
    return env["_build_callable"]


class CompositeWorkflowScorer:
    """Minimal scorer capturing failures."""

    def __init__(self):
        self.failure = False

    def run(self, fn, wf_id, run_id=None):
        try:
            result = fn()
            success = bool(result)
        except Exception:
            self.failure = True
            success = False
        return types.SimpleNamespace(success_rate=1.0 if success else 0.0, roi_gain=0.0)


def test_missing_module_records_failure(caplog):
    build_callable = _load_build_callable()
    with caplog.at_level(logging.ERROR):
        fn = build_callable("no_such_module")
    scorer = CompositeWorkflowScorer()
    result = scorer.run(fn, "wf1")
    assert scorer.failure is True
    assert result.success_rate == 0.0
    assert any("no_such_module" in rec.message for rec in caplog.records)
