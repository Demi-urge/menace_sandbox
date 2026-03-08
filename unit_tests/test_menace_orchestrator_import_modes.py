"""Smoke tests for importing menace_orchestrator in package and script modes."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_PATH = ROOT / "menace_orchestrator.py"


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch, *, package_prefix: str | None) -> None:
    class _Readiness:
        poll_interval = 0.01

        def await_ready(self, timeout: float | None = None) -> None:
            return None

        def probe(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(ready=True)

        def describe(self) -> str:
            return "ready"

    bootstrap_module = _make_module(
        "bootstrap_readiness",
        readiness_signal=lambda: _Readiness(),
        probe_embedding_service=lambda readiness_loop=False: (True, "local-stub"),
    )
    timeout_module = _make_module(
        "bootstrap_timeout_policy",
        resolve_bootstrap_gate_timeout=lambda **kwargs: kwargs.get("fallback_timeout", 180.0),
        compute_prepare_pipeline_component_budgets=lambda *args, **kwargs: {},
        read_bootstrap_heartbeat=lambda *args, **kwargs: None,
    )
    router_module = _make_module(
        "dynamic_path_router",
        resolve_path=lambda path: Path(path),
        get_project_root=lambda: ROOT,
    )
    db_router_module = _make_module("db_router", DBRouter=type("DBRouter", (), {}))

    monkeypatch.setitem(sys.modules, "bootstrap_readiness", bootstrap_module)
    monkeypatch.setitem(sys.modules, "bootstrap_timeout_policy", timeout_module)
    monkeypatch.setitem(sys.modules, "dynamic_path_router", router_module)
    monkeypatch.setitem(sys.modules, "db_router", db_router_module)

    coding_module = _make_module(
        "coding_bot_interface",
        _bootstrap_dependency_broker=lambda *args, **kwargs: object(),
        advertise_bootstrap_placeholder=lambda **kwargs: (object(), object()),
        _resolve_bootstrap_wait_timeout=lambda **kwargs: 180.0,
        _BOOTSTRAP_STATE=types.SimpleNamespace(helper_promotion_callbacks=[]),
        _GLOBAL_BOOTSTRAP_COORDINATOR=object(),
        _current_bootstrap_context=lambda: None,
        claim_bootstrap_dependency_entry=lambda *args, **kwargs: None,
        _peek_owner_promise=lambda: None,
        _resolve_caller_module_name=lambda: "test",
        prepare_pipeline_for_bootstrap=lambda *args, **kwargs: None,
    )
    bootstrap_gate_module = _make_module(
        "bootstrap_gate",
        resolve_bootstrap_placeholders=lambda **kwargs: (
            object(),
            object(),
            types.SimpleNamespace(active_owner=True),
        ),
    )

    names_to_stub = {
        "knowledge_graph": {"KnowledgeGraph": type("KnowledgeGraph", (), {})},
        "advanced_error_management": {
            "AutomatedRollbackManager": type("AutomatedRollbackManager", (), {})
        },
        "self_coding_engine": {"SelfCodingEngine": type("SelfCodingEngine", (), {})},
        "rollback_validator": {"RollbackValidator": type("RollbackValidator", (), {})},
        "oversight_bots": {
            "L1OversightBot": type("L1OversightBot", (), {}),
            "L2OversightBot": type("L2OversightBot", (), {}),
            "L3OversightBot": type("L3OversightBot", (), {}),
            "M1OversightBot": type("M1OversightBot", (), {}),
            "M2OversightBot": type("M2OversightBot", (), {}),
            "M3OversightBot": type("M3OversightBot", (), {}),
            "H1OversightBot": type("H1OversightBot", (), {}),
            "H2OversightBot": type("H2OversightBot", (), {}),
            "H3OversightBot": type("H3OversightBot", (), {}),
        },
        "model_automation_pipeline": {
            "ModelAutomationPipeline": type("ModelAutomationPipeline", (), {}),
            "AutomationResult": type("AutomationResult", (), {}),
        },
        "discrepancy_detection_bot": {
            "DiscrepancyDetectionBot": type("DiscrepancyDetectionBot", (), {})
        },
        "efficiency_bot": {"EfficiencyBot": type("EfficiencyBot", (), {})},
        "neuroplasticity": {
            "Outcome": type("Outcome", (), {}),
            "PathwayDB": type("PathwayDB", (), {}),
            "PathwayRecord": type("PathwayRecord", (), {}),
        },
        "ad_integration": {"AdIntegration": type("AdIntegration", (), {})},
        "watchdog": {
            "Watchdog": type("Watchdog", (), {}),
            "ContextBuilder": type("ContextBuilder", (), {}),
        },
        "error_bot": {"ErrorDB": type("ErrorDB", (), {})},
        "resource_allocation_optimizer": {"ROIDB": type("ROIDB", (), {})},
        "data_bot": {"MetricsDB": type("MetricsDB", (), {})},
        "trending_scraper": {"TrendingScraper": type("TrendingScraper", (), {})},
        "self_learning_service": {"main": lambda: None},
        "strategic_planner": {"StrategicPlanner": type("StrategicPlanner", (), {})},
        "strategy_prediction_bot": {
            "StrategyPredictionBot": type("StrategyPredictionBot", (), {})
        },
        "autoscaler": {"Autoscaler": type("Autoscaler", (), {})},
        "trend_predictor": {"TrendPredictor": type("TrendPredictor", (), {})},
        "identity_seeder": {"seed_identity": lambda: None},
        "session_vault": {"SessionVault": type("SessionVault", (), {})},
        "cognition_layer": {
            "build_cognitive_context": lambda *args, **kwargs: {},
            "log_feedback": lambda *args, **kwargs: None,
        },
    }

    monkeypatch.setitem(sys.modules, "coding_bot_interface", coding_module)
    monkeypatch.setitem(sys.modules, "menace_sandbox.bootstrap_gate", bootstrap_gate_module)

    if package_prefix:
        package = _make_module(package_prefix)
        package.__path__ = [str(ROOT)]
        monkeypatch.setitem(sys.modules, package_prefix, package)
        monkeypatch.setitem(sys.modules, f"{package_prefix}.coding_bot_interface", coding_module)
        monkeypatch.setitem(sys.modules, f"{package_prefix}.bootstrap_gate", bootstrap_gate_module)

    for module_name, attrs in names_to_stub.items():
        module = _make_module(module_name, **attrs)
        monkeypatch.setitem(sys.modules, module_name, module)
        if package_prefix:
            monkeypatch.setitem(sys.modules, f"{package_prefix}.{module_name}", module)


def _import_from_path(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ORCHESTRATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_import_as_top_level_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_import_stubs(monkeypatch, package_prefix=None)

    module = _import_from_path("menace_orchestrator")

    assert module is sys.modules["menace_orchestrator"]


def test_import_as_package_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_import_stubs(monkeypatch, package_prefix="menace")

    module = _import_from_path("menace.menace_orchestrator")

    assert module is sys.modules["menace.menace_orchestrator"]
