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
import threading
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


def test_bootstrap_helpers_share_single_promise(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    base = sys.modules.setdefault("menace_sandbox", ModuleType("menace_sandbox"))
    base.__path__ = [str(Path(__file__).resolve().parents[1])]

    prepare_calls: list[dict[str, object]] = []
    promote_calls: list[object | None] = []

    class _FakePromise:
        def __init__(self, pipeline: object, promoter: object):
            self.pipeline = pipeline
            self.promoter = promoter
            self.waiters = 0

        def wait(self):
            self.waiters += 1
            return self.pipeline, self.promoter

    broker_advertises: list[tuple[object | None, object | None, bool]] = []

    class _FakeBroker:
        def __init__(self, pipeline: object):
            self.pipeline = pipeline

        def resolve(self):
            return self.pipeline, getattr(self.pipeline, "manager", None)

        def advertise(self, pipeline=None, sentinel=None, owner: bool | None = None):  # noqa: ANN001
            broker_advertises.append((pipeline, sentinel, bool(owner)))
            if pipeline is not None:
                self.pipeline = pipeline
            return self.pipeline, sentinel

    pipeline = SimpleNamespace(manager=SimpleNamespace(name="sentinel"))
    promise = _FakePromise(pipeline, lambda manager=None: promote_calls.append(manager))
    active_promise: _FakePromise | None = None

    logger = logging.getLogger("bootstrap-integration")

    def _prepare_pipeline_for_bootstrap(**kwargs: object):
        nonlocal active_promise
        prepare_calls.append(kwargs)
        logger.info("prepare_pipeline_for_bootstrap invoked")
        active_promise = promise
        return promise.pipeline, promise.promoter

    class _Coordinator:
        def peek_active(self):
            return active_promise

    broker = _FakeBroker(pipeline)

    cbi = ModuleType("menace_sandbox.coding_bot_interface")
    cbi.__file__ = str(Path(__file__).resolve())
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR = _Coordinator()
    cbi._bootstrap_dependency_broker = lambda: broker
    cbi.advertise_bootstrap_placeholder = (
        lambda *, dependency_broker=None, pipeline=None, manager=None, owner=True: (
            pipeline or dependency_broker.resolve()[0],
            manager or getattr(dependency_broker.resolve()[0], "manager", None),
        )
    )
    cbi.prepare_pipeline_for_bootstrap = _prepare_pipeline_for_bootstrap
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", cbi)

    def _install_helper(module_name: str, attr: str) -> ModuleType:
        module = ModuleType(module_name)

        def _bootstrap_helper():
            barrier.wait()
            with lock:
                promise_candidate = cbi._GLOBAL_BOOTSTRAP_COORDINATOR.peek_active()
                if promise_candidate is not None:
                    return promise_candidate.wait()
                pipeline_candidate, promoter = cbi.prepare_pipeline_for_bootstrap(
                    helper=module_name
                )
            broker.resolve()
            broker.advertise(
                pipeline=pipeline_candidate,
                sentinel=getattr(pipeline_candidate, "manager", None),
                owner=True,
            )
            return pipeline_candidate, promoter

        setattr(module, attr, _bootstrap_helper)
        monkeypatch.setitem(sys.modules, module_name, module)
        return module

    # Vector service package stub
    vector_service_pkg = ModuleType("vector_service")
    vector_service_pkg.__path__ = [str(Path(__file__).resolve().parent)]
    monkeypatch.setitem(sys.modules, "vector_service", vector_service_pkg)

    helpers = (
        ("menace_sandbox.research_aggregator_bot", "bootstrap_runtime"),
        ("menace_sandbox.prediction_manager_bot", "bootstrap_prediction_manager"),
        ("vector_service.vector_database_service", "bootstrap_vector_service"),
        ("startup_health_check", "bootstrap_health_check"),
    )

    lock = threading.Lock()
    barrier = threading.Barrier(len(helpers))
    results: list[tuple[object, object]] = []

    for module_name, attr in helpers:
        _install_helper(module_name, attr)

    def _run_helper(module_name: str, attr: str) -> None:
        mod = importlib.import_module(module_name)
        results.append(getattr(mod, attr)())

    threads = [
        threading.Thread(target=_run_helper, args=helper, daemon=True) for helper in helpers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(prepare_calls) == 1, "bootstrap should run only once"
    assert promise.waiters == len(helpers) - 1, "waiters should attach to active promise"
    assert all(result[0] is pipeline for result in results)
    assert not any("recursion_refused" in record.getMessage() for record in caplog.records)
    prepare_logs = [
        record
        for record in caplog.records
        if "prepare_pipeline_for_bootstrap" in record.getMessage()
    ]
    assert len(prepare_logs) == 1, "prepare_pipeline_for_bootstrap should log once"
    assert broker_advertises, "dependency broker should be exercised during bootstrap"


def test_helper_import_reuses_advertised_placeholder(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    import menace_sandbox.coding_bot_interface as cbi

    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    cbi._REENTRY_ATTEMPTS.clear()

    manager = SimpleNamespace(name="bootstrap-manager")
    promotions: list[object | None] = []

    def _promote(real_manager: object | None) -> None:
        promotions.append(real_manager)

    pipeline = SimpleNamespace(manager=manager, _pipeline_promoter=_promote)

    class _Broker:
        def __init__(self, pipeline: object, sentinel: object):
            self.pipeline = pipeline
            self.sentinel = sentinel
            self.calls: list[tuple[str, object | None, object | None, bool]] = []
            self.active_owner = False

        def resolve(self):
            return self.pipeline, self.sentinel

        def advertise(self, pipeline=None, sentinel=None, owner: bool | None = None):  # noqa: ANN001
            if pipeline is not None:
                self.pipeline = pipeline
            if sentinel is not None:
                self.sentinel = sentinel
            if owner:
                self.active_owner = True
            self.calls.append(("advertise", self.pipeline, self.sentinel, bool(owner)))
            return self.pipeline, self.sentinel

    dependency_broker = _Broker(pipeline, manager)
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: dependency_broker)

    placeholder_pipeline, placeholder_manager = cbi.advertise_bootstrap_placeholder(
        dependency_broker=dependency_broker,
        pipeline=pipeline,
        manager=manager,
        owner=True,
    )

    owner_started = threading.Event()
    release_owner = threading.Event()
    prepare_calls: list[dict[str, object]] = []

    def _prepare_pipeline_for_bootstrap(**kwargs: object):
        owner, promise = cbi._GLOBAL_BOOTSTRAP_COORDINATOR.claim()
        event_logger = logging.getLogger("bootstrap-owner")
        event_logger.info("prepare-pipeline-stub", extra={"owner": owner, "waiters": promise.waiters})
        prepare_calls.append({"owner": owner, "waiters": promise.waiters, "kwargs": kwargs})
        if owner:
            owner_started.set()
            release_owner.wait(timeout=3)
            dependency_broker.advertise(pipeline=pipeline, sentinel=manager, owner=True)
            promise.resolve((pipeline, _promote))
        return promise.wait()

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _prepare_pipeline_for_bootstrap)

    owner_thread = threading.Thread(
        target=cbi.prepare_pipeline_for_bootstrap,
        kwargs={"helper": "owner"},
        daemon=True,
    )
    owner_thread.start()
    owner_started.wait(timeout=3)

    assert dependency_broker.active_owner, "placeholder owner should mark broker active"
    active_promise = cbi._GLOBAL_BOOTSTRAP_COORDINATOR.peek_active()
    assert active_promise is not None
    assert active_promise.waiters == 1

    helper_results: list[dict[str, object]] = []

    def _install_helper():
        module = ModuleType("menace_sandbox.workflow_evolution_helper")

        def _bootstrap_helper():
            broker = cbi._bootstrap_dependency_broker()
            advertised_pipeline, advertised_manager = cbi.advertise_bootstrap_placeholder(
                dependency_broker=broker,
            )
            pipeline_candidate, promoter = cbi.prepare_pipeline_for_bootstrap(
                helper="workflow-evolution-helper",
            )
            helper_results.append(
                {
                    "advertised_pipeline": advertised_pipeline,
                    "advertised_manager": advertised_manager,
                    "pipeline": pipeline_candidate,
                    "promoter": promoter,
                    "resolved": broker.resolve(),
                }
            )
            return pipeline_candidate, promoter

        module.bootstrap_helper = _bootstrap_helper
        module.__file__ = str(Path(__file__))
        monkeypatch.setitem(sys.modules, "menace_sandbox.workflow_evolution_helper", module)
        return module

    helper_module = _install_helper()

    helper_thread = threading.Thread(
        target=lambda: importlib.import_module(helper_module.__name__).bootstrap_helper(),
        daemon=True,
    )
    helper_thread.start()

    release_owner.set()
    owner_thread.join(timeout=3)
    helper_thread.join(timeout=3)

    assert prepare_calls and prepare_calls[0]["owner"] is True
    assert any(not call["owner"] for call in prepare_calls), "helper should reuse active promise"

    assert helper_results, "bootstrap helper should record results"
    helper_result = helper_results[0]
    assert helper_result["pipeline"] is pipeline
    assert helper_result["advertised_pipeline"] is placeholder_pipeline
    assert helper_result["resolved"][0] is pipeline
    assert helper_result["resolved"][1] is manager

    assert promotions == [], "promoter should not run during reuse"
    assert dependency_broker.calls, "dependency broker should be exercised"
    assert len(dependency_broker.calls) >= 2, "placeholder advertise and reuse should both log"

    owner_logs = [
        record for record in caplog.records if record.getMessage() == "prepare-pipeline-stub"
    ]
    assert len(owner_logs) == 2, "owner and helper should both log prepare telemetry"
    assert all(getattr(record, "owner", None) is not None for record in owner_logs)
    assert {record.owner for record in owner_logs} == {True, False}

    assert cbi._REENTRY_ATTEMPTS.get("menace_sandbox.workflow_evolution_helper") is None

    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()


def test_active_promise_short_circuits_recursion(monkeypatch, caplog):
    import coding_bot_interface as cbi

    caplog.set_level(logging.INFO)
    cbi._REENTRY_ATTEMPTS.clear()

    pipeline = SimpleNamespace(name="pipeline")

    class _Promise:
        def __init__(self) -> None:
            self.waiters = 0
            self.pipeline = pipeline

        def wait(self):
            self.waiters += 1
            return self.pipeline, lambda *_: None

    promise = _Promise()
    broker = SimpleNamespace(
        active_pipeline=SimpleNamespace(manager=SimpleNamespace(name="sentinel")),
        active_sentinel=SimpleNamespace(name="sentinel"),
        active_owner=True,
    )
    broker.resolve = lambda: (broker.active_pipeline, broker.active_sentinel)

    def _refuse_advertise(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("advertise should not run during recursion guard")

    broker.advertise = _refuse_advertise

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", SimpleNamespace(peek_active=lambda: promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1, raising=False)

    try:
        result = cbi.prepare_pipeline_for_bootstrap(
            pipeline_cls=type("Pipeline", (), {}),
            context_builder=None,
            bot_registry=None,
            data_bot=None,
        )
    finally:
        try:
            delattr(cbi._BOOTSTRAP_STATE, "depth")
        except AttributeError:
            pass

    assert result[0].name == "pipeline"
    assert promise.waiters == 1
    assert any("caller_module" in getattr(record, "__dict__", {}) for record in caplog.records)


def test_recursion_cap_short_circuits_to_active_promise(monkeypatch, caplog):
    import coding_bot_interface as cbi

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("MENACE_BOOTSTRAP_REENTRY_CAP", "1")
    cbi._REENTRY_ATTEMPTS.clear()
    cbi._PREPARE_CALL_INVOCATIONS.clear()
    cbi._REENTRY_CAP_ALERTS.clear()

    pipeline = SimpleNamespace(name="pipeline")

    class _Promise:
        def __init__(self) -> None:
            self.waiters = 0
            self.pipeline = pipeline

        def wait(self):
            self.waiters += 1
            return self.pipeline, lambda *_: None

    promise = _Promise()
    broker = SimpleNamespace(
        active_pipeline=SimpleNamespace(manager=SimpleNamespace(name="sentinel")),
        active_sentinel=SimpleNamespace(name="sentinel"),
        active_owner=True,
    )
    broker.resolve = lambda: (broker.active_pipeline, broker.active_sentinel)

    def _refuse_advertise(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("advertise should not run during recursion guard")

    broker.advertise = _refuse_advertise

    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", SimpleNamespace(peek_active=lambda: promise))
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi._BOOTSTRAP_STATE, "depth", 1, raising=False)

    first = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=type("Pipeline", (), {}),
        context_builder=None,
        bot_registry=None,
        data_bot=None,
    )
    second = cbi.prepare_pipeline_for_bootstrap(
        pipeline_cls=type("Pipeline", (), {}),
        context_builder=None,
        bot_registry=None,
        data_bot=None,
    )

    assert first[0] is second[0] is pipeline
    assert promise.waiters >= 2
    assert any(value == 2 for value in cbi._REENTRY_ATTEMPTS.values())
