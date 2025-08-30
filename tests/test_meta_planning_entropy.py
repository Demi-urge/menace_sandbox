from types import SimpleNamespace
from pathlib import Path
import ast


src_path = Path(__file__).resolve().parents[1] / "self_improvement" / "meta_planning.py"
tree = ast.parse(src_path.read_text(), filename=str(src_path))
ns: dict[str, object] = {}
ns["SandboxSettings"] = object
ns["WorkflowStabilityDB"] = object
from typing import Any, Mapping
ns["Any"] = Any
ns["Mapping"] = Mapping
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_get_entropy_threshold",
        "_should_encode",
    }:
        mod = ast.Module([node], type_ignores=[])
        exec(compile(mod, str(src_path), "exec"), ns)

_get_entropy_threshold = ns["_get_entropy_threshold"]
_should_encode = ns["_should_encode"]


def test_threshold_from_settings_overrides_history():
    settings = SimpleNamespace(meta_entropy_threshold=0.1)
    db = SimpleNamespace(data={"a": {"entropy": 0.5}})
    assert _get_entropy_threshold(settings, db) == 0.1


def test_threshold_derived_from_history_when_not_set():
    settings = SimpleNamespace(meta_entropy_threshold=None)
    db = SimpleNamespace(data={"a": {"entropy": 0.1}, "b": {"entropy": 0.4}})
    assert _get_entropy_threshold(settings, db) == 0.4


def test_should_encode_respects_threshold():
    record = {"roi_gain": 0.2, "entropy": 0.25}
    assert _should_encode(record, entropy_threshold=0.3)
    assert not _should_encode(record, entropy_threshold=0.2)

