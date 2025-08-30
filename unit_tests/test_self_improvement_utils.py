import ast
import ast
import importlib
import logging
import time
import types
from pathlib import Path
from typing import Any, Callable, Awaitable
from unittest.mock import patch
import asyncio
import random
import inspect

import pytest


def _load_utils():
    src = Path("self_improvement/utils.py").read_text()
    tree = ast.parse(src)
    wanted = {"_load_callable", "_call_with_retries"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(nodes, type_ignores=[])

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def labels(self, **kw):  # pragma: no cover - simple stub
            return types.SimpleNamespace(inc=lambda: setattr(self, "count", self.count + 1))

    counter = Counter()
    ns = {
        "importlib": importlib,
        "logging": logging,
        "time": time,
        "asyncio": asyncio,
        "random": random,
        "inspect": inspect,
        "Callable": Callable,
        "Any": Any,
        "Awaitable": Awaitable,
        "self_improvement_failure_total": counter,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns, counter


def test_load_callable_success():
    utils, counter = _load_utils()
    fn = utils["_load_callable"]("math", "sqrt")
    assert fn(4) == 2
    assert counter.count == 0


def test_load_callable_missing_increments_metric():
    utils, counter = _load_utils()
    with patch("importlib.import_module", side_effect=ImportError):
        with pytest.raises(RuntimeError):
            utils["_load_callable"]("missing", "attr")
    assert counter.count == 1


def test_call_with_retries_records_failure():
    utils, counter = _load_utils()
    utils["time"].sleep = lambda *a, **k: None

    def always_fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        utils["_call_with_retries"](always_fail, retries=2)
    assert counter.count == 1


def test_context_is_attached_to_log_records(caplog):
    utils, counter = _load_utils()
    utils["time"].sleep = lambda *a, **k: None

    def always_fail():
        raise ValueError("boom")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError):
            utils["_call_with_retries"](
                always_fail, retries=1, context={"foo": "bar"}
            )
    assert any(getattr(r, "foo", None) == "bar" for r in caplog.records)
