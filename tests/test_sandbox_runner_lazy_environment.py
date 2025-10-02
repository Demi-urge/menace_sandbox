"""Regression tests for sandbox_runner lazy import behaviour."""

import importlib
import sys
from typing import Dict


_SANDBOX_PREFIX = "sandbox_runner"


def _sandbox_module_names() -> list[str]:
    return [
        name
        for name in sys.modules
        if name == _SANDBOX_PREFIX or name.startswith(f"{_SANDBOX_PREFIX}.")
    ]


def _stash_sandbox_modules() -> Dict[str, object]:
    stashed: Dict[str, object] = {}
    for name in _sandbox_module_names():
        stashed[name] = sys.modules.pop(name)
    return stashed


def _restore_sandbox_modules(stashed: Dict[str, object]) -> None:
    for name in _sandbox_module_names():
        if name not in stashed:
            sys.modules.pop(name)
    sys.modules.update(stashed)


def test_import_test_harness_before_environment(monkeypatch):
    """Importing the test harness before environment should not raise."""

    monkeypatch.delenv("MENACE_LIGHT_IMPORTS", raising=False)
    stashed = _stash_sandbox_modules()
    try:
        harness = importlib.import_module("sandbox_runner.test_harness")
        assert hasattr(harness, "TestHarnessResult")
        env_module = importlib.import_module("sandbox_runner.environment")
        assert hasattr(env_module, "generate_edge_cases")
    finally:
        _restore_sandbox_modules(stashed)
