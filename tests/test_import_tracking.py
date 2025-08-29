import builtins
import contextlib
import sys
import types

import pytest

from tests.test_workflow_sandbox_runner import WorkflowSandboxRunner


def _make_stub_env(tracker):
    env = types.ModuleType("sandbox_runner.environment")

    @contextlib.contextmanager
    def _patched_imports():
        prev = builtins.__import__
        builtins.__import__ = tracker
        try:
            yield
        finally:
            builtins.__import__ = prev

    env._patched_imports = _patched_imports
    return env


def test_import_tracker_restores_builtins():
    original = builtins.__import__

    def tracker(name, globals=None, locals=None, fromlist=(), level=0):
        return original(name, globals, locals, fromlist, level)

    sys.modules["sandbox_runner.environment"] = _make_stub_env(tracker)

    runner = WorkflowSandboxRunner()
    runner.run(lambda: None)

    assert builtins.__import__ is original


def test_import_tracker_restores_on_exception():
    original = builtins.__import__

    def tracker(name, globals=None, locals=None, fromlist=(), level=0):
        return original(name, globals, locals, fromlist, level)

    sys.modules["sandbox_runner.environment"] = _make_stub_env(tracker)

    runner = WorkflowSandboxRunner()

    with pytest.raises(RuntimeError):
        def fail():
            raise RuntimeError("boom")
        runner.run(fail)

    assert builtins.__import__ is original
