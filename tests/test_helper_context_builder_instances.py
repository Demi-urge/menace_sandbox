import ast
import types
from pathlib import Path


def _load_helper_function():
    src = Path(__file__).resolve().parents[1] / "self_coding_manager.py"
    tree = ast.parse(src.read_text())
    fn_node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_manager_generate_helper_with_builder"
    )
    ns: dict[str, object] = {}
    exec("from typing import Any", ns)
    code = compile(ast.Module([fn_node], []), str(src), "exec")
    exec(code, ns)
    return ns["_manager_generate_helper_with_builder"]


def test_distinct_context_builders():
    builders = []

    class DummyBuilder:
        def __init__(self) -> None:
            self.refreshed = False

        def refresh_db_weights(self) -> None:
            self.refreshed = True

    def ensure_fresh_weights(builder):  # pragma: no cover - simple stub
        builder.refresh_db_weights()

    def base_helper(manager, description, **kwargs):  # pragma: no cover - simple stub
        builders.append(kwargs.get("context_builder"))
        return "ok"

    helper_fn = _load_helper_function()
    # Inject dependencies into helper's global namespace
    helper_fn.__globals__.update(
        {
            "ContextBuilder": DummyBuilder,
            "ensure_fresh_weights": ensure_fresh_weights,
            "_BASE_MANAGER_GENERATE_HELPER": base_helper,
        }
    )

    dummy_manager = types.SimpleNamespace()
    helper_fn(dummy_manager, "first")
    helper_fn(dummy_manager, "second")

    assert len(builders) == 2
    assert builders[0] is not builders[1]
    assert builders[0].refreshed and builders[1].refreshed
