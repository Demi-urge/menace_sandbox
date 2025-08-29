import ast
import importlib
import logging
import time
from pathlib import Path
from unittest.mock import patch
from typing import Callable, Any
import types

import pytest


def _load_utils():
    src = Path("self_improvement/utils.py").read_text()
    tree = ast.parse(src)
    wanted = {"_load_callable", "_call_with_retries"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(nodes, type_ignores=[])
    counter = types.SimpleNamespace(labels=lambda **k: types.SimpleNamespace(inc=lambda: None))
    ns = {
        "importlib": importlib,
        "logging": logging,
        "time": time,
        "Callable": Callable,
        "Any": Any,
        "self_improvement_failure_total": counter,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns


def test_missing_dependency_raises_runtime_error():
    utils = _load_utils()
    with patch("importlib.import_module", side_effect=ModuleNotFoundError):
        with pytest.raises(RuntimeError):
            utils["_load_callable"]("mod", "attr")


def test_retry_succeeds_after_transient_failure():
    utils = _load_utils()
    utils["time"].sleep = lambda *a, **k: None
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("boom")
        return "ok"

    assert utils["_call_with_retries"](flaky, retries=3) == "ok"
    assert attempts["count"] == 2


def test_retry_fails_after_all_attempts():
    utils = _load_utils()
    utils["time"].sleep = lambda *a, **k: None

    def always_fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        utils["_call_with_retries"](always_fail, retries=2)

