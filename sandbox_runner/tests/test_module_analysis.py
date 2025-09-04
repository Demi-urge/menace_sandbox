import ast
import subprocess
from types import SimpleNamespace
from typing import Any, Dict
import logging

import pytest
from pathlib import Path
from dynamic_path_router import resolve_path

# Load _analyse_module from cycle.py without importing heavy dependencies
cycle_path = Path(resolve_path("sandbox_runner/cycle.py"))
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
    "Path": Path,
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
    mod = tmp_path / "dummy.py"  # path-ignore
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

    score, signals = analyse_module(SimpleNamespace(repo=tmp_path), "dummy.py")  # path-ignore
    assert score == pytest.approx(0.5)
    assert signals["commit"] == pytest.approx(0.5)
    assert signals["complexity"] == pytest.approx(0.7)
    assert signals["failures"] == pytest.approx(0.3)


def test_analyse_module_custom_weights(monkeypatch, tmp_path):
    mod = tmp_path / "dummy.py"  # path-ignore
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

    settings = SimpleNamespace(
        risk_weight_commit=0.0, risk_weight_complexity=0.0, risk_weight_failures=1.0
    )
    ctx = SimpleNamespace(repo=tmp_path, settings=settings)
    score, signals = analyse_module(ctx, "dummy.py")  # path-ignore
    assert score == pytest.approx(0.3)
    assert signals["commit"] == pytest.approx(0.5)
    assert signals["complexity"] == pytest.approx(0.7)
    assert signals["failures"] == pytest.approx(0.3)
