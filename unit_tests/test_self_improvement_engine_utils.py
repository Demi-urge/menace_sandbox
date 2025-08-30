import ast
import importlib
import logging
import time
from pathlib import Path
from unittest.mock import patch
from typing import Callable, Any, Awaitable
import types
import asyncio
import random
import inspect
from dataclasses import dataclass

import pytest


def _load_utils():
    src = Path("self_improvement/utils.py").read_text()
    tree = ast.parse(src)
    wanted = {"_load_callable", "_call_with_retries"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(nodes, type_ignores=[])
    counter = types.SimpleNamespace(labels=lambda **k: types.SimpleNamespace(inc=lambda: None))

    class _StubSettings:
        def __init__(self) -> None:
            self.retry_optional_dependencies = False
            self.sandbox_retry_delay = 0
            self.sandbox_max_retries = 0

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
        "SandboxSettings": _StubSettings,
        "dataclass": dataclass,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns


def test_missing_dependency_returns_stub():
    utils = _load_utils()
    with patch("importlib.import_module", side_effect=ModuleNotFoundError):
        fn = utils["_load_callable"]("mod", "attr")
        with pytest.raises(RuntimeError):
            fn()


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


def test_load_callable_retry_when_enabled():
    utils = _load_utils()
    utils["time"].sleep = lambda *a, **k: None

    attempts = {"count": 0}

    def side_effect(name):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ModuleNotFoundError
        return types.SimpleNamespace(attr=lambda: "ok")

    with patch("importlib.import_module", side_effect=side_effect):
        utils["SandboxSettings"] = lambda: types.SimpleNamespace(
            retry_optional_dependencies=True,
            sandbox_retry_delay=0,
            sandbox_max_retries=3,
        )
        fn = utils["_load_callable"]("mod", "attr")
        assert fn() == "ok"


def test_retry_fails_after_all_attempts():
    utils = _load_utils()
    utils["time"].sleep = lambda *a, **k: None

    def always_fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        utils["_call_with_retries"](always_fail, retries=2)


def test_async_retry_and_backoff_features():
    utils = _load_utils()
    sleeps: list[float] = []
    utils["time"].sleep = lambda t: sleeps.append(t)
    utils["random"].uniform = lambda a, b: 0.5
    attempts = {"count": 0}

    async def flaky_async():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("boom")
        return "ok"

    assert (
        utils["_call_with_retries"](
            flaky_async, retries=3, delay=1, jitter=1, max_delay=2
        )
        == "ok"
    )
    assert attempts["count"] == 2
    assert sleeps == [1.5]
