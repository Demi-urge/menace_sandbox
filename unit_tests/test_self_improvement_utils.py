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
from dataclasses import dataclass
import subprocess
import sys
import threading
from functools import lru_cache

import pytest


def _load_utils():
    src = Path("self_improvement/utils.py").read_text()
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

    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def labels(self, **kw):  # pragma: no cover - simple stub
            return types.SimpleNamespace(inc=lambda: setattr(self, "count", self.count + 1))

    counter = Counter()

    class _StubSettings:
        def __init__(self) -> None:
            self.retry_optional_dependencies = False
            self.sandbox_retry_delay = 0
            self.sandbox_max_retries = 0
            self.menace_offline_install = False

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
        "subprocess": subprocess,
        "sys": sys,
        "threading": threading,
        "lru_cache": lru_cache,
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


def test_load_callable_missing_returns_stub_and_records_metric():
    utils, counter = _load_utils()
    with patch("subprocess.run", side_effect=Exception) as sp, patch(
        "importlib.import_module", side_effect=ImportError
    ):
        fn = utils["_load_callable"]("missing", "attr", allow_install=True)
        with pytest.raises(RuntimeError) as ei:
            fn()
    assert counter.count == 1
    assert fn.error.module == "missing"
    assert fn.error.install_attempted is True
    assert "pip install missing" in str(ei.value)
    assert utils["_load_callable"].diagnostics["install_attempts"] == 1
    assert sp.called


def test_no_install_without_flag():
    utils, counter = _load_utils()
    with patch("subprocess.run") as sp, patch(
        "importlib.import_module", side_effect=ImportError
    ):
        fn = utils["_load_callable"]("missing", "attr")
        with pytest.raises(RuntimeError):
            fn()
    assert fn.error.install_attempted is False
    assert utils["_load_callable"].diagnostics["install_attempts"] == 0
    sp.assert_not_called()


def test_load_callable_retry_succeeds_when_enabled():
    utils, counter = _load_utils()
    utils["time"].sleep = lambda *a, **k: None

    attempts = {"count": 0}

    def side_effect(name):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ImportError
        return types.SimpleNamespace(attr=lambda: "ok")

    with patch("subprocess.check_call", side_effect=Exception), patch(
        "importlib.import_module", side_effect=side_effect
    ):
        utils["SandboxSettings"] = lambda: types.SimpleNamespace(
            retry_optional_dependencies=True,
            sandbox_retry_delay=0,
            sandbox_max_retries=3,
            menace_offline_install=False,
        )
        fn = utils["_load_callable"]("mod", "attr")
        assert fn() == "ok"
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
