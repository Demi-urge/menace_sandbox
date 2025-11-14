import importlib
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _simple_class(label: str):
    class _Stub:
        def __init__(self, *args, **kwargs):
            self._label = label

    _Stub.__name__ = label
    return _Stub


@pytest.fixture()
def orchestrator_module(monkeypatch):
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.syspath_prepend(str(ROOT))
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "menace", menace_pkg)

    def _install(name: str, attrs: dict[str, object]) -> None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        if name.startswith("menace."):
            short = name.split(".", 1)[1]
            monkeypatch.setitem(sys.modules, short, module)
        else:
            monkeypatch.setitem(sys.modules, f"menace.{name}", module)

    _install(
        "dynamic_path_router",
        {
            "resolve_path": lambda path: Path(path),
            "get_project_root": lambda: ROOT,
        },
    )
    _install("db_router", {"DBRouter": _simple_class("DBRouter"), "GLOBAL_ROUTER": None})
    _install("menace.coding_bot_interface", {"prepare_pipeline_for_bootstrap": lambda **_: (None, lambda *_a, **_k: None)})
    _install("menace.knowledge_graph", {"KnowledgeGraph": _simple_class("KnowledgeGraph")})
    _install("menace.advanced_error_management", {"AutomatedRollbackManager": _simple_class("AutomatedRollbackManager")})
    _install("menace.self_coding_engine", {"SelfCodingEngine": _simple_class("SelfCodingEngine")})
    _install("menace.rollback_validator", {"RollbackValidator": _simple_class("RollbackValidator")})
    _install(
        "menace.oversight_bots",
        {name: _simple_class(name) for name in [
            "L1OversightBot",
            "L2OversightBot",
            "L3OversightBot",
            "M1OversightBot",
            "M2OversightBot",
            "M3OversightBot",
            "H1OversightBot",
            "H2OversightBot",
            "H3OversightBot",
        ]},
    )
    _install(
        "menace.model_automation_pipeline",
        {
            "ModelAutomationPipeline": _simple_class("ModelAutomationPipeline"),
            "AutomationResult": _simple_class("AutomationResult"),
        },
    )
    _install("menace.discrepancy_detection_bot", {"DiscrepancyDetectionBot": _simple_class("DiscrepancyDetectionBot")})
    _install("menace.efficiency_bot", {"EfficiencyBot": _simple_class("EfficiencyBot")})
    _install(
        "menace.neuroplasticity",
        {
            "Outcome": _simple_class("Outcome"),
            "PathwayDB": _simple_class("PathwayDB"),
            "PathwayRecord": _simple_class("PathwayRecord"),
        },
    )
    _install("menace.ad_integration", {"AdIntegration": _simple_class("AdIntegration")})
    _install(
        "menace.watchdog",
        {
            "Watchdog": _simple_class("Watchdog"),
            "ContextBuilder": _simple_class("ContextBuilder"),
        },
    )
    _install("menace.error_bot", {"ErrorDB": _simple_class("ErrorDB")})
    _install("menace.resource_allocation_optimizer", {"ROIDB": _simple_class("ROIDB")})
    _install("menace.data_bot", {"MetricsDB": _simple_class("MetricsDB")})
    _install("menace.trending_scraper", {"TrendingScraper": _simple_class("TrendingScraper")})
    _install("menace.self_learning_service", {"main": lambda *a, **k: None})
    _install("menace.strategic_planner", {"StrategicPlanner": _simple_class("StrategicPlanner")})
    _install("menace.strategy_prediction_bot", {"StrategyPredictionBot": _simple_class("StrategyPredictionBot")})
    _install("menace.autoscaler", {"Autoscaler": _simple_class("Autoscaler")})
    _install("menace.trend_predictor", {"TrendPredictor": _simple_class("TrendPredictor")})
    _install("menace.identity_seeder", {"seed_identity": lambda *a, **k: None})
    _install("menace.session_vault", {"SessionVault": _simple_class("SessionVault")})
    _install(
        "menace.cognition_layer",
        {
            "build_cognitive_context": lambda *a, **k: {},
            "log_feedback": lambda *a, **k: None,
        },
    )

    sys.modules.pop("menace.menace_orchestrator", None)
    return importlib.import_module("menace.menace_orchestrator")


def test_menace_orchestrator_promotes_bootstrap_pipeline(orchestrator_module, monkeypatch, caplog):
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    pipeline = types.SimpleNamespace(manager="sentinel")
    promote_calls: list[object] = []

    def _fake_prepare(**kwargs):
        assert kwargs["pipeline_cls"] is orchestrator_module.ModelAutomationPipeline

        def _promote(manager):
            promote_calls.append(manager)
            pipeline.manager = manager

        return pipeline, _promote

    monkeypatch.setattr(
        orchestrator_module,
        "prepare_pipeline_for_bootstrap",
        _fake_prepare,
        raising=False,
    )
    router = types.SimpleNamespace(menace_id="test", get_connection=lambda *a, **k: None)
    orchestrator = orchestrator_module.MenaceOrchestrator(context_builder=builder, router=router)

    assert orchestrator.pipeline is pipeline
    assert callable(orchestrator.pipeline_promoter)
    orchestrator.promote_pipeline_manager("manager")
    assert pipeline.manager == "manager"
    assert promote_calls == ["manager"]
    assert orchestrator.pipeline_promoter is None
    assert "re-entrant initialisation depth" not in caplog.text
