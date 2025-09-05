from types import SimpleNamespace
from pathlib import Path
import ast
import pytest
from typing import Any, Mapping, Sequence
from statistics import fmean
from self_improvement.baseline_tracker import BaselineTracker


src_path = Path(__file__).resolve().parents[1] / "self_improvement" / "meta_planning.py"  # path-ignore
tree = ast.parse(src_path.read_text(), filename=str(src_path))
ns: dict[str, object] = {}
ns["SandboxSettings"] = object
ns["BaselineTracker"] = object
ns["Any"] = Any
ns["Mapping"] = Mapping
ns["Sequence"] = Sequence
ns["fmean"] = fmean
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
    "cfg_value, base, std, dev, expected",
    [
        (0.1, 0.2, 0.3, 1.0, 0.1),
        (None, 0.1, 0.05, 2.0, 0.2),
        (None, 0.2, 0.0, 1.0, 0.2),
    ],
)
def test_get_entropy_threshold(cfg_value, base, std, dev, expected):
    settings = SimpleNamespace(meta_entropy_threshold=cfg_value, entropy_deviation=dev)
    tracker = SimpleNamespace(get=lambda name: base, std=lambda name: std)

    assert _get_entropy_threshold(settings, tracker) == pytest.approx(expected)


def test_should_encode_respects_threshold():
    tracker = BaselineTracker(window=3)
    tracker.update(roi=0.0, pass_rate=1.0, entropy=0.3)
    record = {"roi_gain": 0.2, "entropy": 0.5, "failures": 0}
    tracker.update(roi=record["roi_gain"], pass_rate=1.0, entropy=record["entropy"])
    ok_high, _ = _should_encode(record, tracker, entropy_threshold=0.6)
    ok_low, reason_low = _should_encode(record, tracker, entropy_threshold=0.2)
    assert ok_high
    assert not ok_low and reason_low == "entropy_spike"


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
