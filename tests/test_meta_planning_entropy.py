from types import SimpleNamespace
from pathlib import Path
import ast
import pytest
from typing import Any, Mapping, Sequence
from statistics import fmean
from importlib import import_module


src_path = Path(__file__).resolve().parents[1] / "self_improvement" / "meta_planning.py"
tree = ast.parse(src_path.read_text(), filename=str(src_path))
ns: dict[str, object] = {}
ns["SandboxSettings"] = object
ns["WorkflowStabilityDB"] = object
ns["Any"] = Any
ns["Mapping"] = Mapping
ns["Sequence"] = Sequence
ns["DEFAULT_ENTROPY_THRESHOLD"] = 0.2
ns["fmean"] = fmean
ns["import_module"] = import_module
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_get_entropy_threshold",
        "_should_encode",
    }:
        mod = ast.Module([node], type_ignores=[])
        exec(compile(mod, str(src_path), "exec"), ns)
    if isinstance(node, ast.ClassDef) and node.name == "_FallbackPlanner":
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name == "_score":
                mod = ast.Module([sub], type_ignores=[])
                exec(compile(mod, str(src_path), "exec"), ns)

_get_entropy_threshold = ns["_get_entropy_threshold"]
_should_encode = ns["_should_encode"]
_fp_score = ns["_score"]


@pytest.mark.parametrize(
    "cfg_value, history, db_data, expected",
    [
        (0.1, [{"code_diversity": 0.2, "token_complexity": 0.4}], {"a": {"entropy": 0.5}}, 0.1),
        (
            None,
            [
                {"code_diversity": 0.1, "token_complexity": 0.3},
                {"code_diversity": 0.2, "token_complexity": 0.5},
            ],
            {},
            0.275,
        ),
        (None, [], {"a": {"entropy": 0.1}, "b": {"entropy": 0.4}}, 0.25),
        (None, [], {}, 0.2),
    ],
)
def test_get_entropy_threshold(cfg_value, history, db_data, expected):
    settings = SimpleNamespace(meta_entropy_threshold=cfg_value)
    db = SimpleNamespace(data=db_data)

    stub = SimpleNamespace(
        get_alignment_metrics=lambda cfg: {"entropy_history": history}
    )
    _get_entropy_threshold.__globals__["import_module"] = lambda name: stub

    assert _get_entropy_threshold(settings, db) == pytest.approx(expected)


def test_should_encode_respects_threshold():
    record = {"roi_gain": 0.2, "entropy": 0.5}
    assert _should_encode(record, entropy_threshold=0.6)
    assert not _should_encode(record, entropy_threshold=0.2)


def test_fallback_score_penalizes_delta(monkeypatch):
    class P:
        roi_weight = 1.0
        domain_transition_penalty = 0.0
        entropy_weight = 1.0
        stability_weight = 0.0
        entropy_threshold = 0.2

        def _domain(self, wid: str) -> str:
            return wid.split(".", 1)[0]

        _score = _fp_score

    planner = P()
    s1 = planner._score(["a"], 1.0, 0.2, 0)
    s2 = planner._score(["a"], 1.0, 0.6, 0)
    assert s2 < s1
