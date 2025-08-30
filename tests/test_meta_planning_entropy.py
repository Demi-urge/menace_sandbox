from types import SimpleNamespace
from pathlib import Path
import ast
import pytest
from typing import Any, Mapping


src_path = Path(__file__).resolve().parents[1] / "self_improvement" / "meta_planning.py"
tree = ast.parse(src_path.read_text(), filename=str(src_path))
ns: dict[str, object] = {}
ns["SandboxSettings"] = object
ns["WorkflowStabilityDB"] = object
ns["Any"] = Any
ns["Mapping"] = Mapping
ns["DEFAULT_ENTROPY_THRESHOLD"] = 0.2
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_get_entropy_threshold",
        "_should_encode",
    }:
        mod = ast.Module([node], type_ignores=[])
        exec(compile(mod, str(src_path), "exec"), ns)

_get_entropy_threshold = ns["_get_entropy_threshold"]
_should_encode = ns["_should_encode"]


@pytest.mark.parametrize(
    "cfg_value, db_data, expected",
    [
        (0.1, {"a": {"entropy": 0.5}}, 0.1),
        (None, {"a": {"entropy": 0.1}, "b": {"entropy": 0.4}}, 0.4),
        (None, {}, 0.2),
    ],
)
def test_get_entropy_threshold(cfg_value, db_data, expected):
    settings = SimpleNamespace(meta_entropy_threshold=cfg_value)
    db = SimpleNamespace(data=db_data)
    assert _get_entropy_threshold(settings, db) == expected


def test_should_encode_respects_threshold():
    record = {"roi_gain": 0.2, "entropy": 0.25}
    assert _should_encode(record, entropy_threshold=0.3)
    assert not _should_encode(record, entropy_threshold=0.2)
