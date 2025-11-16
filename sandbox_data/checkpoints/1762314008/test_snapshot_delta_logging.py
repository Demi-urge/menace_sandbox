import ast
import json
import types
import pytest
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

    def _log(
        prompt,
        success,
        exec_result,
        roi_meta,
        prompt_id=None,
        failure_reason=None,
        sandbox_metrics=None,
        commit_hash=None,
    ):  # pragma: no cover
        log_entries.append(
            {
                "prompt": prompt,
                "success": success,
                "exec_result": exec_result,
                "roi_meta": roi_meta,
                "failure_reason": failure_reason,
                "sandbox_metrics": sandbox_metrics,
                "commit_hash": commit_hash,
            }
        )

    from pathlib import Path
    from typing import Sequence

    ns = {
        "_data_dir": lambda: tmp_path,
        "json": json,
        "log_prompt_attempt": _log,
        "Path": Path,
        "Sequence": Sequence,
    }
    exec(compile(module, str(resolve_path("self_improvement/engine.py")), "exec"), ns)
    func = ns["_record_snapshot_delta"]

    class Stub(types.SimpleNamespace):
        def __init__(self):
            super().__init__(logger=types.SimpleNamespace(exception=lambda *a, **k: None))
            self.confidence_updater = types.SimpleNamespace(update=lambda d: updates.append(d))

    return func, Stub()


@pytest.mark.parametrize(
    "delta,failed,reason",
    [
        ({"roi": -1.0}, ["roi"], "roi_drop"),
        ({"sandbox_score": -0.1}, ["sandbox_score"], "score_drop"),
        ({"entropy": 0.1}, ["entropy"], "entropy_regression"),
        ({"tests_passed": False}, ["tests_failed"], "tests_failed"),
        (
            {"roi": -1.0, "sandbox_score": -0.2},
            ["roi", "sandbox_score"],
            "roi_drop",
        ),
    ],
)
def test_record_snapshot_delta_failures(tmp_path, delta, failed, reason):
    logs: list[dict] = []
    updates: list[dict] = []
    func, eng = _load_record_snapshot_delta(tmp_path, logs, updates)
    func(eng, "p", "d", delta)
    path = tmp_path / "snapshots" / "deltas.jsonl"
    assert json.loads(path.read_text().strip()) == delta
    assert logs and not logs[0]["success"]
    assert logs[0]["failure_reason"] == reason
    assert logs[0]["sandbox_metrics"]["failed_metrics"] == failed
    for k, v in delta.items():
        assert logs[0]["sandbox_metrics"][k] == v
    assert not updates


def test_record_snapshot_delta_success(tmp_path):
    logs: list[dict] = []
    updates: list[dict] = []
    func, eng = _load_record_snapshot_delta(tmp_path, logs, updates)
    delta = {"roi": 1.0, "entropy": -0.5, "sandbox_score": 0.1}
    func(eng, "p", "d", delta)
    assert logs and logs[0]["success"]
    assert logs[0]["failure_reason"] is None
    assert logs[0]["sandbox_metrics"] is None
    assert updates and updates[0] == delta
