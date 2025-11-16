import ast
import importlib
import logging
import time
from unittest.mock import patch
from dynamic_path_router import resolve_path
from typing import Callable, Any, Awaitable
import types
import asyncio
import random
import inspect
from dataclasses import dataclass
import threading
from functools import lru_cache
import subprocess
import sys

import pytest


def _load_utils():
    src = resolve_path("self_improvement/utils.py").read_text()
    tree = ast.parse(src)
    wanted_funcs = {"_load_callable", "_call_with_retries", "_import_callable"}
    wanted_assigns = {"_diagnostics_lock", "_diagnostics"}
    nodes: list[ast.stmt] = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in wanted_funcs:
            nodes.append(n)
        elif isinstance(n, ast.Assign):
            targets = {t.id for t in n.targets if isinstance(t, ast.Name)}
            if targets & wanted_assigns:
                nodes.append(n)
    module = ast.Module(nodes, type_ignores=[])
    counter = types.SimpleNamespace(labels=lambda **k: types.SimpleNamespace(inc=lambda: None))

    class _StubSettings:
        def __init__(self) -> None:
            self.retry_optional_dependencies = False
            self.install_optional_dependencies = False
            self.optional_service_versions: dict[str, str] = {}
            self.sandbox_retry_delay = 0
            self.sandbox_max_retries = 0
            self.menace_offline_install = False
            self.sandbox_retry_backoff_multiplier = 1.0
            self.sandbox_retry_jitter = 0.0

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
        "threading": threading,
        "lru_cache": lru_cache,
        "subprocess": subprocess,
        "sys": sys,
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


def test_async_retry_and_backoff_features():
    utils = _load_utils()
    sleeps: list[float] = []
    utils["time"].sleep = lambda t: sleeps.append(t)
    utils["random"].uniform = lambda a, b: b
    attempts = {"count": 0}

    async def flaky_async():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("boom")
        return "ok"

    assert (
        utils["_call_with_retries"](
            flaky_async, retries=3, delay=1, jitter=0.1, max_delay=1.5
        )
        == "ok"
    )
    assert attempts["count"] == 3
    assert sleeps == [1.1, 1.5]


def test_async_context_uses_asyncio_sleep():
    utils = _load_utils()
    sleeps: list[float] = []

    async def fake_sleep(t: float) -> None:
        sleeps.append(t)

    utils["asyncio"].sleep = fake_sleep
    attempts = {"count": 0}

    async def flaky_async():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("boom")
        return "ok"

    async def runner():
        return await utils["_call_with_retries"](flaky_async, retries=2, delay=1)

    result = asyncio.run(runner())
    assert result == "ok"
    assert attempts["count"] == 2
    assert sleeps == [1]
