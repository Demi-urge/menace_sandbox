"""Ensure bootstrap clients reuse advertised placeholders.

The genetic algorithm, enhancement, and bot-creation helpers should reuse
dependency-broker placeholders or active promises instead of spawning new
``prepare_pipeline_for_bootstrap`` calls when a bootstrap is already in flight.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
for _opt in (
    "menace_sandbox.truth_adapter",
    "menace_sandbox.foresight_tracker",
    "menace_sandbox.upgrade_forecaster",
):
    sys.modules.setdefault(_opt, ModuleType(_opt))

_pkg = sys.modules.setdefault("menace_sandbox", ModuleType("menace_sandbox"))
_pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
_pkg.__file__ = str(Path(__file__).resolve().parents[1] / "__init__.py")

class _StubBroker:
    def __init__(self, pipeline: object) -> None:
        self.pipeline = pipeline
        self.calls: list[tuple[str, object, object, bool]] = []
        self.active_pipeline: object | None = pipeline
        self.active_sentinel: object | None = getattr(pipeline, "manager", None)

    def resolve(self):
        return self.active_pipeline, self.active_sentinel

    def advertise(self, pipeline=None, sentinel=None, owner: bool | None = None):  # noqa: ANN001
        self.calls.append(("advertise", pipeline, sentinel, bool(owner)))
        if pipeline is not None:
            self.active_pipeline = pipeline
        if sentinel is not None:
            self.active_sentinel = sentinel
        return self.active_pipeline, self.active_sentinel


def _install_stub_module(monkeypatch, name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_clients_reuse_broker_placeholder(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    for mod in (
        "menace_sandbox.coding_bot_interface",
        "menace_sandbox.genetic_algorithm_bot",
        "menace_sandbox.enhancement_bot",
        "menace_sandbox.bot_creation_bot",
    ):
        sys.modules.pop(mod, None)

    import menace_sandbox.coding_bot_interface as cbi

    manager = SimpleNamespace(name="manager")
    promote_calls: list[object] = []

    def _promoter(real_manager: object | None) -> None:
        promote_calls.append(real_manager)

    pipeline = SimpleNamespace(manager=manager, _pipeline_promoter=_promoter)
    broker = _StubBroker(pipeline)
    promise = SimpleNamespace(wait=lambda: (pipeline, _promoter))

    _install_stub_module(
        monkeypatch,
        "context_builder_util",
        create_context_builder=lambda: "context",
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.context_builder_util",
        create_context_builder=lambda: "context",
    )

    threshold_stub = SimpleNamespace(roi_drop=1, error_increase=2, test_failure_increase=3)
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.self_coding_thresholds",
        get_thresholds=lambda *_: threshold_stub,
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.threshold_service",
        ThresholdService=type("ThresholdService", (), {}),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.shared_evolution_orchestrator",
        get_orchestrator=lambda *_a, **_k: SimpleNamespace(),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.self_coding_manager",
        SelfCodingManager=type("SelfCodingManager", (), {}),
        internalize_coding_bot=lambda *_a, **_k: manager,
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.self_coding_engine",
        SelfCodingEngine=lambda *_a, **_k: SimpleNamespace(),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.code_database",
        CodeDB=lambda *_a, **_k: SimpleNamespace(),
        CodeRecord=type("CodeRecord", (), {}),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.gpt_memory",
        GPTMemoryManager=lambda *_a, **_k: SimpleNamespace(),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.bot_registry",
        BotRegistry=lambda *_a, **_k: SimpleNamespace(),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.data_bot",
        DataBot=lambda *_a, **_k: SimpleNamespace(),
        MetricsDB=type("MetricsDB", (), {}),
        persist_sc_thresholds=lambda *_a, **_k: None,
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.shared.model_pipeline_core",
        ModelAutomationPipeline=type("ModelAutomationPipeline", (), {}),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.model_automation_pipeline",
        ModelAutomationPipeline=type("ModelAutomationPipeline", (), {}),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.menace_memory_manager",
        MenaceMemoryManager=lambda *_a, **_k: SimpleNamespace(),
    )

    _install_stub_module(
        monkeypatch, "menace_sandbox.bot_planning_bot", BotPlanningBot=object, PlanningTask=object
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.bot_development_bot",
        BotDevelopmentBot=object,
        BotSpec=object,
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.deployment_bot",
        DeploymentBot=object,
        DeploymentSpec=object,
    )
    _install_stub_module(monkeypatch, "menace_sandbox.error_bot", ErrorBot=object)
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.scalability_assessment_bot",
        ScalabilityAssessmentBot=object,
    )
    _install_stub_module(monkeypatch, "menace_sandbox.safety_monitor", SafetyMonitor=object)
    _install_stub_module(
        monkeypatch, "menace_sandbox.prediction_manager_bot", PredictionManager=object
    )
    _install_stub_module(monkeypatch, "menace_sandbox.learning_engine", LearningEngine=object)
    _install_stub_module(
        monkeypatch, "menace_sandbox.evolution_analysis_bot", EvolutionAnalysisBot=object
    )
    _install_stub_module(monkeypatch, "menace_sandbox.stripe_billing_router")
    _install_stub_module(monkeypatch, "menace_sandbox.workflow_evolution_bot", WorkflowEvolutionBot=object)
    _install_stub_module(monkeypatch, "menace_sandbox.trending_scraper", TrendingScraper=object)
    _install_stub_module(monkeypatch, "menace_sandbox.admin_bot_base", AdminBotBase=object)
    _install_stub_module(monkeypatch, "menace_sandbox.roi_tracker", ROITracker=object)
    _install_stub_module(
        monkeypatch, "menace_sandbox.menace_sanity_layer", fetch_recent_billing_issues=lambda *_a, **_k: []
    )
    _install_stub_module(monkeypatch, "menace_sandbox.dynamic_path_router", path_for_prompt=lambda *_a, **_k: "")
    _install_stub_module(monkeypatch, "menace_sandbox.intent_clusterer", IntentClusterer=object)
    _install_stub_module(monkeypatch, "menace_sandbox.universal_retriever", UniversalRetriever=object)

    _install_stub_module(
        monkeypatch,
        "menace_sandbox.entry_pipeline_loader",
        load_pipeline_class=lambda: type("ModelAutomationPipeline", (), {}),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.database_manager",
        DB_PATH="/tmp/db",
        update_model=lambda *_a, **_k: None,
        process_idea=lambda *_a, **_k: None,
    )
    _install_stub_module(
        monkeypatch,
        "vector_service.cognition_layer",
        CognitionLayer=type("CognitionLayer", (), {}),
    )
    _install_stub_module(
        monkeypatch,
        "vector_service.context_builder",
        ContextBuilder=type("ContextBuilder", (), {}),
    )

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", SimpleNamespace(peek_active=lambda: promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi, "get_active_bootstrap_pipeline", lambda: (pipeline, manager))
    monkeypatch.setattr(
        cbi,
        "advertise_bootstrap_placeholder",
        lambda dependency_broker, pipeline=None, manager=None: (
            pipeline or dependency_broker.active_pipeline,
            manager or dependency_broker.active_sentinel,
        ),
    )
    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda *_, **__: {"active": True})
    monkeypatch.setattr(
        cbi, "_current_bootstrap_context", lambda: SimpleNamespace(pipeline=pipeline, manager=manager)
    )

    prepare_mock = mock.Mock(side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run"))
    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", prepare_mock)

    for target in (
        "menace_sandbox.genetic_algorithm_bot",
        "menace_sandbox.enhancement_bot",
        "menace_sandbox.bot_creation_bot",
    ):
        sys.modules.pop(target, None)

    ga_bot = importlib.import_module("menace_sandbox.genetic_algorithm_bot")
    enhancement_bot = importlib.import_module("menace_sandbox.enhancement_bot")
    creation_bot = importlib.import_module("menace_sandbox.bot_creation_bot")

    ga_manager = ga_bot._manager_proxy()

    assert ga_manager is manager
    assert enhancement_bot.pipeline is pipeline
    assert creation_bot.pipeline is pipeline
    assert promote_calls == [manager]
    assert any(call[0] == "advertise" for call in broker.calls)
    prepare_mock.assert_not_called()
    assert any(
        record.getMessage() == "bot_creation.bootstrap.reuse" for record in caplog.records
    ), "bot creation should log reuse instead of new prepare"
    assert not any(
        "prepare_pipeline_for_bootstrap" in record.getMessage() for record in caplog.records
    )
