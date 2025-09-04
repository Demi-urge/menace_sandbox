import ast
import json
import types
from dynamic_path_router import resolve_path


def _load_record_snapshot_delta(tmp_path, log_entries, updates):
    source = resolve_path("self_improvement/engine.py").read_text()
    mod = ast.parse(source)
    func_node = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "SelfImprovementEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_record_snapshot_delta":
                    func_node = item
                    break
    assert func_node is not None
    module = ast.Module(body=[func_node], type_ignores=[])

    def _log(prompt, success, exec_result, roi_meta, prompt_id=None):  # pragma: no cover
        log_entries.append(
            {
                "prompt": prompt,
                "success": success,
                "exec_result": exec_result,
                "roi_meta": roi_meta,
            }
        )

    ns = {
        "_data_dir": lambda: tmp_path,
        "json": json,
        "log_prompt_attempt": _log,
    }
    exec(compile(module, str(resolve_path("self_improvement/engine.py")), "exec"), ns)
    func = ns["_record_snapshot_delta"]

    class Stub(types.SimpleNamespace):
        def __init__(self):
            super().__init__(logger=types.SimpleNamespace(exception=lambda *a, **k: None))
            self.confidence_updater = types.SimpleNamespace(update=lambda d: updates.append(d))

    return func, Stub()


def test_record_snapshot_delta_regression(tmp_path):
    logs: list[dict] = []
    updates: list[dict] = []
    func, eng = _load_record_snapshot_delta(tmp_path, logs, updates)
    delta = {"roi": -1.0, "entropy": 0.0}
    func(eng, "p", "d", delta)
    path = tmp_path / "snapshots" / "deltas.jsonl"
    assert json.loads(path.read_text().strip()) == delta
    assert logs and not logs[0]["success"]
    assert logs[0]["exec_result"]["diff"] == "d"
    assert not updates


def test_record_snapshot_delta_success(tmp_path):
    logs: list[dict] = []
    updates: list[dict] = []
    func, eng = _load_record_snapshot_delta(tmp_path, logs, updates)
    delta = {"roi": 1.0, "entropy": 0.5}
    func(eng, "p", "d", delta)
    assert logs and logs[0]["success"]
    assert updates and updates[0] == delta
