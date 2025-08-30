import ast
import pathlib
import subprocess
from types import SimpleNamespace
from typing import Any, Dict
import logging

import pytest

# Load _analyse_module from cycle.py without importing heavy dependencies
cycle_path = pathlib.Path(__file__).resolve().parents[1] / "cycle.py"
source = cycle_path.read_text()
module = ast.parse(source)
func_src = None
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_analyse_module":
        func_src = ast.get_source_segment(source, node)
        break
if func_src is None:  # pragma: no cover - sanity
    raise RuntimeError("_analyse_module not found")

globals_dict: Dict[str, Any] = {
    "Path": pathlib.Path,
    "subprocess": subprocess,
    "router": SimpleNamespace(),
    "mi_visit": None,
    "logger": logging.getLogger(__name__),
    "Any": Any,
    "Dict": Dict,
}
exec(func_src, globals_dict)
analyse_module = globals_dict["_analyse_module"]


def test_analyse_module_weighted_score(monkeypatch, tmp_path):
    mod = tmp_path / "dummy.py"
    mod.write_text("print('hi')")

    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **k: "c1\nc2\nc3\nc4\nc5\n"
    )
    analyse_module.__globals__["mi_visit"] = lambda code, *a: 30

    class DummyCur:
        def fetchone(self):
            return (3,)

    class DummyConn:
        def execute(self, *a, **k):
            return DummyCur()

        def close(self):
            pass

    class DummyRouter:
        def get_connection(self, *_):
            return DummyConn()

    analyse_module.__globals__["router"] = DummyRouter()

    score, signals = analyse_module(SimpleNamespace(repo=tmp_path), "dummy.py")
    assert score == pytest.approx(0.5)
    assert signals["commit"] == pytest.approx(0.5)
    assert signals["complexity"] == pytest.approx(0.7)
    assert signals["failures"] == pytest.approx(0.3)
