# flake8: noqa
"""Unit tests for the autonomous sandbox launch helper.

The real ``self_improvement.engine`` module has a very heavy import graph.
For these tests we extract just the helper-related definitions via ``ast`` and
execute them in a scratch module.  This keeps the tests fast while still using
the production implementation of :func:`launch_autonomous_sandbox`.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import threading
import types
from pathlib import Path

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def _load_engine_subset() -> types.ModuleType:
    source = Path("self_improvement/engine.py").read_text()
    module_ast = ast.parse(source, filename="self_improvement/engine.py")

    needed_funcs = {"_is_truthy_env", "launch_autonomous_sandbox"}
    needed_assigns = {"_MANUAL_LAUNCH_TRIGGERED", "_AUTONOMOUS_SANDBOX_LAUNCH_ENV"}
    filtered: list[ast.stmt] = []

    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & needed_assigns:
                filtered.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in needed_funcs:
            filtered.append(node)

    subset_module = ast.Module(body=filtered, type_ignores=[])
    compiled = compile(subset_module, "self_improvement/engine.py", "exec")

    module = types.ModuleType("self_improvement.engine_test")
    module.importlib = importlib
    module.os = os
    module.threading = threading
    module.logger = types.SimpleNamespace(exception=lambda *a, **k: None)
    module._qfe_log = lambda *a, **k: None
    exec(compiled, module.__dict__)
    return module


@pytest.fixture
def engine_module() -> types.ModuleType:
    return _load_engine_subset()


@pytest.fixture(autouse=True)
def reset_launch_state(engine_module, monkeypatch):
    monkeypatch.setattr(engine_module, "_MANUAL_LAUNCH_TRIGGERED", False, raising=False)
    monkeypatch.delenv(engine_module._AUTONOMOUS_SANDBOX_LAUNCH_ENV, raising=False)


def test_launch_autonomous_sandbox_respects_env(engine_module, monkeypatch):
    calls: list[str] = []

    def _unexpected(name: str):
        calls.append(name)
        raise AssertionError("run_autonomous should not be imported when flag is unset")

    monkeypatch.setattr(engine_module.importlib, "import_module", _unexpected)
    result = engine_module.launch_autonomous_sandbox(run_args=["--noop"], background=False)

    assert result is None
    assert not calls
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is False


def test_launch_autonomous_sandbox_env_trigger(engine_module, monkeypatch):
    monkeypatch.setenv(engine_module._AUTONOMOUS_SANDBOX_LAUNCH_ENV, "1")

    captured: list[list[str]] = []
    module = types.SimpleNamespace(main=lambda argv=None: captured.append(list(argv or [])))

    def _import(name: str):
        if name in {"menace_sandbox.run_autonomous", "run_autonomous"}:
            return module
        raise ImportError(name)

    monkeypatch.setattr(engine_module.importlib, "import_module", _import)

    result = engine_module.launch_autonomous_sandbox(run_args=["--flag"], background=False)

    assert result is None
    assert captured == [["--flag"]]
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is True


def test_launch_autonomous_sandbox_background_thread(engine_module, monkeypatch):
    monkeypatch.setenv(engine_module._AUTONOMOUS_SANDBOX_LAUNCH_ENV, "1")

    captured: list[list[str]] = []

    def _main(argv=None):
        captured.append(list(argv or []))

    module = types.SimpleNamespace(main=_main)

    def _import(name: str):
        if name in {"menace_sandbox.run_autonomous", "run_autonomous"}:
            return module
        raise ImportError(name)

    monkeypatch.setattr(engine_module.importlib, "import_module", _import)

    thread = engine_module.launch_autonomous_sandbox(run_args=["--bg"], background=True)

    assert thread is not None
    thread.join(timeout=1)
    assert captured == [["--bg"]]
    assert engine_module._MANUAL_LAUNCH_TRIGGERED is True


def test_autonomous_bootstrap_invokes_launch(engine_module, monkeypatch):
    # Provide the subset module to autonomous_bootstrap before importing it.
    sys.modules["self_improvement.engine"] = engine_module
    bootstrap = importlib.import_module("autonomous_bootstrap")

    monkeypatch.setattr(
        bootstrap,
        "bootstrap_environment",
        lambda settings, verifier: settings,
    )
    monkeypatch.setattr(bootstrap, "init_self_improvement", lambda settings: None)
    monkeypatch.setattr(
        bootstrap,
        "discover_and_register_coding_bots",
        lambda registry, orchestrator: None,
    )

    class _DummyThread:
        def __init__(self):
            self.started = False
            self.join_called = False

        def start(self):
            self.started = True

        def join(self):
            self.join_called = True

        def stop(self):  # pragma: no cover - not used in this test
            pass

    dummy_thread = _DummyThread()
    monkeypatch.setattr(
        bootstrap,
        "start_self_improvement_cycle",
        lambda *_a, **_k: dummy_thread,
    )
    monkeypatch.setattr(bootstrap, "BotRegistry", lambda: object())

    launch_calls: list[dict[str, object]] = []

    def _launch(**kwargs):
        launch_calls.append(kwargs)
        return None

    monkeypatch.setattr(bootstrap, "launch_autonomous_sandbox", _launch)

    exit_code = bootstrap.main()

    assert exit_code == 0
    assert dummy_thread.started is True
    assert dummy_thread.join_called is True
    assert launch_calls
    assert launch_calls[-1] == {"background": True, "force": True}
