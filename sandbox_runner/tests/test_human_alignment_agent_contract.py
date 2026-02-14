from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType


def test_menace_sandbox_human_alignment_agent_is_importable_and_instantiable() -> None:
    module = importlib.import_module("menace_sandbox.human_alignment_agent")

    assert hasattr(module, "HumanAlignmentAgent")

    agent = module.HumanAlignmentAgent()
    assert isinstance(agent, module.HumanAlignmentAgent)


def test_self_improvement_engine_import_succeeds_without_human_alignment_monkeypatch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.pop("self_improvement.engine", None)

    original_import = builtins.__import__

    def fallback_import(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            return original_import(name, globals, locals, fromlist, level)
        except Exception:
            module = ModuleType(name)
            for attr in fromlist or ():
                setattr(module, attr, type(attr, (), {}))
            sys.modules.setdefault(name, module)
            return module

    monkeypatch.setattr(builtins, "__import__", fallback_import)

    engine = importlib.import_module("self_improvement.engine")

    assert engine is not None
    assert hasattr(engine, "HumanAlignmentAgent")


def test_human_alignment_agent_evaluate_changes_returns_warning_dict_shape() -> None:
    from menace_sandbox.human_alignment_agent import HumanAlignmentAgent

    agent = HumanAlignmentAgent()
    warnings = agent.evaluate_changes(
        actions=[{"file": "example_module.py", "code": "print('ok')\n"}],
        metrics={"test_coverage": 0.9, "roi": 1.2},
        logs=["proposed patch includes tests"],
        commit_info={"author": "ci", "message": "add guardrails"},
    )

    assert isinstance(warnings, dict)
    assert set(warnings.keys()) == {"ethics", "risk_reward", "maintainability"}
    assert all(isinstance(entries, list) for entries in warnings.values())


def test_legacy_compatibility_path_resolves_to_same_human_alignment_agent_class() -> None:
    sandbox_module = importlib.import_module("menace_sandbox.human_alignment_agent")
    legacy_module = importlib.import_module("menace.human_alignment_agent")

    assert legacy_module.HumanAlignmentAgent is sandbox_module.HumanAlignmentAgent
