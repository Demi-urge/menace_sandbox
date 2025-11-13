import logging
import os
import sys
import types
from pathlib import Path

import pytest

os.environ["MENACE_LOCAL_DB_PATH"] = "/tmp/menace_local.db"
os.environ["MENACE_SHARED_DB_PATH"] = "/tmp/menace_shared.db"

import sandbox_runner


class DummyBuilder:
    def refresh_db_weights(self) -> None:
        pass


class DummyManager:
    def __init__(self) -> None:
        self.summary_payload: dict[str, object] = {}
        self.context_builder = None
        self.bot_name = "ServiceSupervisor"
        self.engine = types.SimpleNamespace(last_added_modules=[], added_modules=[])

    def auto_run_patch(self, path: Path, description: str):
        return {
            "summary": self.summary_payload,
            "patch_id": 1,
            "commit": "abc123",
            "result": None,
        }


def _install_stub_module(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module


def _noop(*args, **kwargs):
    return None


class _StubService:
    def __init__(self, *args, **kwargs):
        pass

    def run_continuous(self, *args, **kwargs):
        pass


class _StubOrchestrator:
    def __init__(self, *args, **kwargs):
        pass

    def create_oversight(self, *args, **kwargs):
        pass

    def start_scheduled_jobs(self):
        pass

    def stop_scheduled_jobs(self):
        pass

    def run_cycle(self, *args, **kwargs):
        return {}


class _StubHealer:
    def __init__(self, *args, **kwargs):
        self.graph = types.SimpleNamespace(add_telemetry_event=lambda *a, **k: None)
        self.heal = lambda *a, **k: None


class _StubKnowledgeGraph:
    def add_crash_trace(self, *args, **kwargs):
        pass

    def add_telemetry_event(self, *args, **kwargs):
        pass


class _StubAutoscaler:
    def scale(self, *args, **kwargs):
        pass


class _StubBootstrapper:
    def bootstrap(self):
        pass


class _StubQuickFixEngine:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass


class _StubBotRegistry:
    def __init__(self, *args, **kwargs):
        pass


class _StubDataBot:
    def __init__(self, *args, **kwargs):
        pass


class _StubContextBuilder:
    def refresh_db_weights(self):
        pass


def _identity_decorator(*args, **kwargs):
    def _wrap(cls):
        return cls

    return _wrap


def _default_prepare_pipeline_for_bootstrap(**_):
    return types.SimpleNamespace(), lambda *_a, **_k: None


_stub_master = types.ModuleType("menace_sandbox.menace_master")
_stub_master._init_unused_bots = lambda: None
sys.modules.setdefault("menace_sandbox.menace_master", _stub_master)

_install_stub_module(
    "menace_sandbox.menace_orchestrator", MenaceOrchestrator=_StubOrchestrator
)
_install_stub_module("menace_sandbox.microtrend_service", MicrotrendService=_StubService)
_install_stub_module(
    "menace_sandbox.self_evaluation_service", SelfEvaluationService=_StubService
)
_install_stub_module("menace_sandbox.self_learning_service", main=_noop)
_install_stub_module(
    "menace_sandbox.cross_model_scheduler", ModelRankingService=_StubService
)
_install_stub_module(
    "menace_sandbox.dependency_update_service",
    DependencyUpdateService=_StubService,
)
_install_stub_module(
    "menace_sandbox.advanced_error_management",
    SelfHealingOrchestrator=_StubHealer,
    AutomatedRollbackManager=lambda *a, **k: types.SimpleNamespace(),
)
_install_stub_module(
    "menace_sandbox.knowledge_graph", KnowledgeGraph=_StubKnowledgeGraph
)
_install_stub_module(
    "menace_sandbox.chaos_monitoring_service", ChaosMonitoringService=_StubService
)
_install_stub_module(
    "menace_sandbox.model_evaluation_service", ModelEvaluationService=_StubService
)
_install_stub_module(
    "menace_sandbox.secret_rotation_service", SecretRotationService=_StubService
)
_install_stub_module(
    "menace_sandbox.environment_bootstrap", EnvironmentBootstrapper=_StubBootstrapper
)
_install_stub_module(
    "menace_sandbox.external_dependency_provisioner",
    ExternalDependencyProvisioner=_StubService,
)
_install_stub_module(
    "menace_sandbox.dependency_watchdog", DependencyWatchdog=_StubService
)
_install_stub_module(
    "menace_sandbox.environment_restoration_service",
    EnvironmentRestorationService=_StubService,
)
_install_stub_module("menace_sandbox.startup_checks", run_startup_checks=_noop)
_install_stub_module("menace_sandbox.autoscaler", Autoscaler=_StubAutoscaler)
_install_stub_module(
    "menace_sandbox.unified_update_service", UnifiedUpdateService=_StubService
)
_install_stub_module(
    "menace_sandbox.self_test_service", SelfTestService=_StubService
)
_install_stub_module(
    "menace_sandbox.auto_escalation_manager",
    AutoEscalationManager=lambda *a, **k: types.SimpleNamespace(handle=_noop),
)
_install_stub_module(
    "menace_sandbox.self_coding_manager",
    PatchApprovalPolicy=lambda *a, **k: types.SimpleNamespace(),
    _manager_generate_helper_with_builder=lambda *a, **k: None,
    internalize_coding_bot=lambda *a, **k: DummyManager(),
)
_install_stub_module("menace_sandbox.error_bot", ErrorDB=lambda *a, **k: types.SimpleNamespace())
_install_stub_module(
    "menace_sandbox.self_coding_engine",
    SelfCodingEngine=lambda *a, **k: types.SimpleNamespace(),
)
_install_stub_module("menace_sandbox.code_database", CodeDB=lambda *a, **k: object())
_install_stub_module(
    "menace_sandbox.menace_memory_manager", MenaceMemoryManager=lambda *a, **k: object()
)
_install_stub_module(
    "menace_sandbox.model_automation_pipeline",
    ModelAutomationPipeline=lambda *a, **k: types.SimpleNamespace(),
)
_install_stub_module(
    "menace_sandbox.quick_fix_engine", QuickFixEngine=_StubQuickFixEngine
)
_install_stub_module(
    "menace_sandbox.shared_event_bus", event_bus=types.SimpleNamespace()
)
_install_stub_module("menace_sandbox.bot_registry", BotRegistry=_StubBotRegistry)
_install_stub_module(
    "menace_sandbox.data_bot", DataBot=_StubDataBot, persist_sc_thresholds=_noop
)
_install_stub_module(
    "menace_sandbox.self_coding_thresholds",
    get_thresholds=lambda name: types.SimpleNamespace(
        roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
    ),
)
_install_stub_module(
    "menace_sandbox.coding_bot_interface",
    self_coding_managed=_identity_decorator,
    prepare_pipeline_for_bootstrap=_default_prepare_pipeline_for_bootstrap,
)
_install_stub_module(
    "menace_sandbox.shared_evolution_orchestrator",
    get_orchestrator=lambda *a, **k: types.SimpleNamespace(),
)
vector_pkg = types.ModuleType("vector_service")
vector_pkg.__path__ = []
sys.modules.setdefault("vector_service", vector_pkg)
_install_stub_module(
    "vector_service.context_builder", ContextBuilder=_StubContextBuilder
)
_install_stub_module(
    "context_builder_util", create_context_builder=lambda: DummyBuilder()
)

import menace_sandbox.service_supervisor as ss


@pytest.fixture(autouse=True)
def _stub_sandbox_runner(monkeypatch):
    monkeypatch.setattr(
        sandbox_runner,
        "try_integrate_into_workflows",
        lambda *a, **k: None,
        raising=False,
    )


def test_constructor_promotes_pipeline(monkeypatch, tmp_path):
    ss.bus = types.SimpleNamespace()
    ss.registry = types.SimpleNamespace()
    ss.data_bot = types.SimpleNamespace()
    promotions: list[DummyManager] = []

    def _fake_prepare_pipeline_for_bootstrap(**_):
        pipeline = types.SimpleNamespace()

        def _promote(manager):
            promotions.append(manager)

        return pipeline, _promote

    manager = DummyManager()
    monkeypatch.setattr(ss, "prepare_pipeline_for_bootstrap", _fake_prepare_pipeline_for_bootstrap)
    monkeypatch.setattr(
        ss,
        "SelfHealingOrchestrator",
        lambda *a, **k: types.SimpleNamespace(
            graph=types.SimpleNamespace(add_telemetry_event=lambda *args, **kwargs: None)
        ),
        raising=False,
    )
    monkeypatch.setattr(ss, "AutomatedRollbackManager", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(ss, "PatchApprovalPolicy", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        ss,
        "AutoEscalationManager",
        lambda *a, **k: types.SimpleNamespace(handle=lambda *args, **kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(ss, "CodeDB", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(ss, "MenaceMemoryManager", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(
        ss,
        "SelfCodingEngine",
        lambda *a, **k: types.SimpleNamespace(last_added_modules=[], added_modules=[]),
        raising=False,
    )
    monkeypatch.setattr(ss, "get_orchestrator", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        ss,
        "get_thresholds",
        lambda name: types.SimpleNamespace(
            roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
        ),
        raising=False,
    )
    monkeypatch.setattr(ss, "persist_sc_thresholds", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ss, "internalize_coding_bot", lambda *a, **k: manager, raising=False)
    monkeypatch.setattr(ss, "ErrorDB", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        ss,
        "QuickFixEngine",
        lambda *a, **k: types.SimpleNamespace(run=lambda *args, **kwargs: None),
        raising=False,
    )

    supervisor = ss.ServiceSupervisor(
        context_builder=DummyBuilder(),
        log_path=str(tmp_path / "supervisor.log"),
        restart_log=str(tmp_path / "restart.log"),
    )

    assert promotions == [manager]
    assert getattr(supervisor, "_pipeline_promoter") is None


def test_deploy_patch_requires_self_tests(monkeypatch, tmp_path):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.context_builder = DummyBuilder()
    supervisor.approval_policy = types.SimpleNamespace()
    rollback_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _record_rollback(*args, **kwargs):
        rollback_calls.append((args, kwargs))

    supervisor.rollback_mgr = types.SimpleNamespace(auto_rollback=_record_rollback)
    logger = logging.getLogger("ServiceSupervisorTest")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    supervisor.logger = logger

    manager = DummyManager()

    promotions: list[DummyManager] = []

    def _fake_prepare_pipeline_for_bootstrap(**_):
        pipeline = types.SimpleNamespace()

        def _promote(real_manager):
            promotions.append(real_manager)

        return pipeline, _promote

    monkeypatch.setattr(ss, "prepare_pipeline_for_bootstrap", _fake_prepare_pipeline_for_bootstrap)
    monkeypatch.setattr(ss, "SelfCodingEngine", lambda *a, **k: manager.engine, raising=False)
    monkeypatch.setattr(ss, "ModelAutomationPipeline", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(ss, "DataBot", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(ss, "CapitalManagementBot", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(ss, "get_orchestrator", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(ss, "internalize_coding_bot", lambda *a, **k: manager, raising=False)
    monkeypatch.setattr(
        ss,
        "get_thresholds",
        lambda name: types.SimpleNamespace(
            roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
        ),
        raising=False,
    )
    monkeypatch.setattr(ss, "persist_sc_thresholds", lambda *a, **k: None, raising=False)

    ss.bus = types.SimpleNamespace()
    ss.registry = types.SimpleNamespace()
    ss.data_bot = types.SimpleNamespace()

    target = tmp_path / "svc.py"
    target.write_text("print('hi')\n")

    supervisor.deploy_patch(target, "desc")
    assert promotions == [manager]
    assert rollback_calls, "rollback should be triggered on failure"

    manager.summary_payload = {"self_tests": {"failed": 0}}
    supervisor.deploy_patch(target, "desc")
    assert promotions == [manager, manager]
