import ast
import importlib
import logging
import time
import types
from dynamic_path_router import resolve_path
from typing import Any, Callable, Awaitable
from unittest.mock import patch
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
    wanted_funcs = {
        "_load_callable",
        "_call_with_retries",
        "_import_callable",
        "clear_import_cache",
    }
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

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def labels(self, **kw):  # pragma: no cover - simple stub
            return types.SimpleNamespace(inc=lambda: setattr(self, "count", self.count + 1))

    counter = Counter()

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
    return ns, counter


def test_load_callable_success_and_cache():
    utils, counter = _load_utils()
    module = types.SimpleNamespace(attr=lambda: "ok")
    with patch("importlib.import_module", return_value=module) as mod:
        fn1 = utils["_load_callable"]("mod", "attr")
        fn2 = utils["_load_callable"]("mod", "attr")
        assert fn1 is fn2
    assert mod.call_count == 1
    assert utils["_load_callable"].diagnostics["cache_hits"] == 1
    assert utils["_load_callable"].diagnostics["cache_misses"] == 1
    assert counter.count == 0


def test_clear_import_cache_resets_cache_and_diagnostics():
    utils, _ = _load_utils()
    module = types.SimpleNamespace(attr=lambda: "ok")
    with patch("importlib.import_module", return_value=module) as mod:
        utils["_load_callable"]("mod", "attr")
        utils["_load_callable"]("mod", "attr")
        assert mod.call_count == 1
        utils["clear_import_cache"]()
        utils["_load_callable"]("mod", "attr")
        assert mod.call_count == 2
        assert utils["_load_callable"].diagnostics["cache_hits"] == 0
        assert utils["_load_callable"].diagnostics["cache_misses"] == 1


def test_load_callable_missing_raises_runtime_error_and_records_metric():
    utils, counter = _load_utils()
    with patch("importlib.import_module", side_effect=ImportError):
        with pytest.raises(RuntimeError) as ei:
            utils["_load_callable"]("missing", "attr")
    assert counter.count == 1
    assert "pip install missing" in str(ei.value)


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


def test_async_context_executes_without_nested_loops():
    utils, _ = _load_utils()
    sleeps: list[float] = []

    async def fake_sleep(t: float) -> None:
        sleeps.append(t)

    utils["asyncio"].sleep = fake_sleep
    attempts = {"count": 0}

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("boom")
        return "ok"

    async def runner():
        return await utils["_call_with_retries"](flaky, retries=2, delay=1)

    result = asyncio.run(runner())
    assert result == "ok"
    assert attempts["count"] == 2
    assert sleeps == [1]
