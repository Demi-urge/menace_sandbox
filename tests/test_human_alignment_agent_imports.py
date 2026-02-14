from __future__ import annotations

import importlib
import importlib.machinery
import sys
from types import ModuleType

import pytest

from sandbox_settings import SandboxSettings


def test_self_improvement_api_import_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    for module_name in list(sys.modules):
        if module_name.startswith("self_improvement") or module_name.startswith(
            "menace_sandbox.self_improvement"
        ):
            sys.modules.pop(module_name, None)

    import builtins

    orig_import = builtins.__import__
    allowed_prefixes = {
        "_",
        "asyncio",
        "ast",
        "collections",
        "concurrent",
        "contextlib",
        "datetime",
        "functools",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "pickle",
        "queue",
        "re",
        "sqlite3",
        "sys",
        "tempfile",
        "threading",
        "time",
        "traceback",
        "types",
        "typing",
        "warnings",
        "yaml",
    }

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top in allowed_prefixes:
            return orig_import(name, globals, locals, fromlist, level)
        try:
            return orig_import(name, globals, locals, fromlist, level)
        except Exception:
            module = ModuleType(name)
            for attr in fromlist or ():
                setattr(module, attr, type(attr, (), {}))
            sys.modules.setdefault(name, module)
            return module

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    imported_module = importlib.import_module("menace_sandbox.self_improvement.api")
    assert imported_module is not None


def test_menace_sandbox_human_alignment_agent_import() -> None:
    import menace_sandbox.human_alignment_agent as haa

    assert hasattr(haa, "HumanAlignmentAgent")


def test_menace_human_alignment_agent_compatibility() -> None:
    import menace.human_alignment_agent as legacy

    assert hasattr(legacy, "HumanAlignmentAgent")


def test_self_improvement_engine_import_isolated_from_human_alignment_agent_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    filelock_stub = ModuleType("filelock")

    class DummyLock:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    filelock_stub.FileLock = DummyLock
    filelock_stub.Timeout = type("Timeout", (Exception,), {})
    monkeypatch.setitem(sys.modules, "filelock", filelock_stub)

    for module_name in list(sys.modules):
        if module_name == "self_improvement.engine" or module_name.startswith("self_improvement"):
            sys.modules.pop(module_name, None)

    imported_engine = importlib.import_module("self_improvement.engine")

    assert imported_engine is not None
    agent = imported_engine.HumanAlignmentAgent(
        settings=SandboxSettings(improvement_warning_threshold=1),
    )
    warnings = agent.evaluate_changes(
        workflow_changes=[{"file": "engine_module.py", "code": "print('hello')\n"}],
        metrics={},
        logs=[],
    )

    assert set(warnings.keys()) == {"ethics", "risk_reward", "maintainability"}


def test_human_alignment_agent_evaluate_changes_logs_when_threshold_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from menace_sandbox.human_alignment_agent import HumanAlignmentAgent

    log_calls: list[tuple[tuple, dict]] = []

    def capture_log_violation(*args, **kwargs):
        log_calls.append((args, kwargs))

    import menace_sandbox.human_alignment_agent as haa

    monkeypatch.setattr(haa, "log_violation", capture_log_violation)

    agent = HumanAlignmentAgent(
        settings=SandboxSettings(improvement_warning_threshold=1),
    )

    warnings = agent.evaluate_changes(
        workflow_changes=[{"file": "module.py", "code": "print('hello')\n"}],
        metrics={},
        logs=[],
    )

    assert set(warnings.keys()) == {"ethics", "risk_reward", "maintainability"}
    assert log_calls, "Expected at least one violation log when threshold conditions are met"


def test_human_alignment_agent_reward_override_keyword_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from menace.human_alignment_agent import HumanAlignmentAgent

    log_calls: list[tuple[tuple, dict]] = []

    def capture_log_violation(*args, **kwargs):
        log_calls.append((args, kwargs))

    import menace.human_alignment_agent as haa

    monkeypatch.setattr(haa, "log_violation", capture_log_violation)

    agent = HumanAlignmentAgent(settings=SandboxSettings(improvement_warning_threshold=0))
    warnings = agent.evaluate_changes(
        actions=[{"file": "module.py", "code": "print('hello')\n"}],
        metrics={},
        logs=[],
        reward_override=True,
    )

    assert "ethics" in warnings
    assert set(warnings.keys()) == {"ethics", "risk_reward", "maintainability"}
    assert log_calls
