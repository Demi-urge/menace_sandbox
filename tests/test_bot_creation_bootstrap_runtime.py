"""Bootstrap reuse safeguards for :mod:`bot_creation_bot`."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


def _install_stub_module(monkeypatch, name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _seed_shared_stubs(monkeypatch):
    """Install the shared stubs bot_creation_bot expects at import time."""

    _install_stub_module(
        monkeypatch,
        "context_builder_util",
        create_context_builder=lambda: "context-builder",
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.context_builder_util",
        create_context_builder=lambda: "context-builder",
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.self_coding_thresholds",
        get_thresholds=lambda *_a, **_k: SimpleNamespace(
            roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0
        ),
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
        internalize_coding_bot=lambda *_a, **_k: SimpleNamespace(name="manager"),
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
        "menace_sandbox.menace_memory_manager",
        MenaceMemoryManager=lambda *_a, **_k: SimpleNamespace(),
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
        "menace_sandbox.entry_pipeline_loader",
        load_pipeline_class=lambda: type("ModelAutomationPipeline", (), {}),
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.bot_planning_bot",
        BotPlanningBot=object,
        PlanningTask=object,
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
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.error_bot",
        ErrorBot=object,
    )
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
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.workflow_evolution_bot",
        WorkflowEvolutionBot=object,
    )
    _install_stub_module(
        monkeypatch, "menace_sandbox.trending_scraper", TrendingScraper=object
    )
    _install_stub_module(monkeypatch, "menace_sandbox.admin_bot_base", AdminBotBase=object)
    _install_stub_module(monkeypatch, "menace_sandbox.roi_tracker", ROITracker=object)
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.menace_sanity_layer",
        fetch_recent_billing_issues=lambda *_a, **_k: [],
    )
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.dynamic_path_router",
        path_for_prompt=lambda *_a, **_k: "",
    )
    _install_stub_module(monkeypatch, "menace_sandbox.intent_clusterer", IntentClusterer=object)
    _install_stub_module(
        monkeypatch, "menace_sandbox.universal_retriever", UniversalRetriever=object
    )
    vector_service_pkg = ModuleType("vector_service")
    vector_service_pkg.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore
    monkeypatch.setitem(sys.modules, "vector_service", vector_service_pkg)
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
    _install_stub_module(
        monkeypatch,
        "menace_sandbox.database_manager",
        DB_PATH="/tmp/db",
        update_model=lambda *_a, **_k: None,
        process_idea=lambda *_a, **_k: None,
    )


def test_bot_creation_reuses_active_bootstrap_promise(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    for target in (
        "menace_sandbox.bot_creation_bot",
        "menace_sandbox.bootstrap_placeholder",
        "menace_sandbox.coding_bot_interface",
    ):
        sys.modules.pop(target, None)

    import menace_sandbox.coding_bot_interface as cbi

    pipeline = SimpleNamespace(
        manager=SimpleNamespace(name="sentinel", bootstrap_placeholder=True),
        bootstrap_placeholder=True,
    )
    broker_calls: list[tuple[str, object | None, object | None, bool]] = []

    class _Broker:
        def resolve(self):
            return pipeline, pipeline.manager

        def advertise(self, pipeline=None, sentinel=None, owner: bool | None = None):  # noqa: ANN001
            broker_calls.append(("advertise", pipeline, sentinel, bool(owner)))
            return pipeline, sentinel

    promise_waits: list[int] = []

    class _Promise:
        def wait(self):
            promise_waits.append(len(promise_waits))
            return pipeline, mock.Mock(name="promote")

    _seed_shared_stubs(monkeypatch)

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", SimpleNamespace(peek_active=lambda: _Promise()))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: _Broker())
    monkeypatch.setattr(
        cbi,
        "advertise_bootstrap_placeholder",
        lambda *, dependency_broker=None, pipeline=None, manager=None, owner=True: (
            pipeline or SimpleNamespace(bootstrap_placeholder=True, manager=manager),
            manager or SimpleNamespace(bootstrap_placeholder=True),
        ),
    )
    monkeypatch.setattr(cbi, "get_active_bootstrap_pipeline", lambda: (None, None))
    monkeypatch.setattr(
        cbi,
        "prepare_pipeline_for_bootstrap",
        mock.Mock(side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")),
    )
    monkeypatch.setattr(cbi, "self_coding_managed", lambda **_k: (lambda cls: cls))
    monkeypatch.setattr(cbi, "normalise_manager_arg", lambda manager, *_a, **_k: manager)

    module = importlib.import_module("menace_sandbox.bot_creation_bot")

    assert module.pipeline is pipeline
    assert module.manager is not None
    assert promise_waits == [0]
    module.prepare_pipeline_for_bootstrap.assert_not_called()
    reuse_logs = [r for r in caplog.records if r.getMessage() == "bot_creation.bootstrap.reuse"]
    assert len(reuse_logs) == 1
    assert any(call[0] == "advertise" for call in broker_calls)


def test_bot_creation_reuses_broker_placeholder_during_bootstrap(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    for target in (
        "menace_sandbox.bot_creation_bot",
        "menace_sandbox.bootstrap_placeholder",
        "menace_sandbox.coding_bot_interface",
    ):
        sys.modules.pop(target, None)

    import menace_sandbox.coding_bot_interface as cbi

    pipeline_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)

    broker_records: list[tuple[str, object | None, object | None, bool]] = []

    class _Broker:
        active_pipeline = pipeline_placeholder
        active_sentinel = sentinel_placeholder

        def resolve(self):
            return self.active_pipeline, self.active_sentinel

        def advertise(self, pipeline=None, sentinel=None, owner: bool | None = None):  # noqa: ANN001
            broker_records.append(("advertise", pipeline, sentinel, bool(owner)))
            if pipeline is not None:
                self.active_pipeline = pipeline
            if sentinel is not None:
                self.active_sentinel = sentinel
            return self.active_pipeline, self.active_sentinel

    promise = SimpleNamespace(wait=lambda: (pipeline_placeholder, mock.Mock(name="promote")))

    _seed_shared_stubs(monkeypatch)

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", SimpleNamespace(peek_active=lambda: promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: _Broker())
    monkeypatch.setattr(
        cbi,
        "advertise_bootstrap_placeholder",
        lambda *, dependency_broker=None, pipeline=None, manager=None, owner=True: (
            pipeline or pipeline_placeholder,
            manager or sentinel_placeholder,
        ),
    )
    monkeypatch.setattr(cbi, "get_active_bootstrap_pipeline", lambda: (None, None))
    monkeypatch.setattr(
        cbi,
        "prepare_pipeline_for_bootstrap",
        mock.Mock(side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")),
    )
    monkeypatch.setattr(cbi, "self_coding_managed", lambda **_k: (lambda cls: cls))
    monkeypatch.setattr(cbi, "normalise_manager_arg", lambda manager, *_a, **_k: manager)

    module = importlib.import_module("menace_sandbox.bot_creation_bot")

    assert module.pipeline is pipeline_placeholder
    module.prepare_pipeline_for_bootstrap.assert_not_called()
    reuse_logs = [r for r in caplog.records if r.getMessage() == "bot_creation.bootstrap.reuse"]
    assert len(reuse_logs) == 1
    assert any(call[0] == "advertise" for call in broker_records)
