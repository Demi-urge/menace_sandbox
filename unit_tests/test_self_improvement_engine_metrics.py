import ast
from pathlib import Path
from dynamic_path_router import resolve_path


def _load_data_dir(sandbox_dir):
    src = resolve_path("self_improvement/__init__.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_data_dir")
    module = ast.Module([fn], type_ignores=[])
    ns = {"SandboxSettings": lambda: type("S", (), {"sandbox_data_dir": str(sandbox_dir)})(), "Path": Path}
    exec(compile(module, "<ast>", "exec"), ns)
    return ns["_data_dir"]


def test_metrics_db_path(tmp_path):
    _data_dir = _load_data_dir(tmp_path)

    class DummyMetricsDB:
        def __init__(self, path):
            self.path = path

    class SelfImprovementEngine:
        def __init__(self):
            self.metrics_db = None
            if self.metrics_db is None:
                try:
                    data_dir = _data_dir()
                    self.metrics_db = DummyMetricsDB(data_dir / "relevancy_metrics.db")
                except Exception:
                    self.metrics_db = None

    eng = SelfImprovementEngine()
    assert eng.metrics_db.path == tmp_path / "relevancy_metrics.db"

