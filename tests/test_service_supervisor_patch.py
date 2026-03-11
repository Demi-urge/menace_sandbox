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
    "menace_sandbox.chaos_monitoring_service",
    ChaosMonitoringService=_StubService,
    resolve_context_builder=lambda *a, **k: None,
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
    _bootstrap_dependency_broker=lambda: types.SimpleNamespace(
        resolve=lambda: (None, None)
    ),
    _current_bootstrap_context=lambda: None,
    _BOOTSTRAP_STATE=types.SimpleNamespace(),
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

vector_pkg_local = types.ModuleType("menace_sandbox.vector_service")
vector_pkg_local.__path__ = []
sys.modules.setdefault("menace_sandbox.vector_service", vector_pkg_local)
_install_stub_module(
    "menace_sandbox.vector_service.context_builder", ContextBuilder=_StubContextBuilder
)

_install_stub_module(
    "context_builder_util", create_context_builder=lambda: DummyBuilder()
)
_install_stub_module(
    "menace_sandbox.context_builder_util", create_context_builder=lambda: DummyBuilder()
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


def test_deploy_patch_blocks_bootstrap_construction(monkeypatch, tmp_path):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.context_builder = DummyBuilder()
    supervisor.approval_policy = types.SimpleNamespace()
    supervisor.rollback_mgr = types.SimpleNamespace(auto_rollback=lambda *_a, **_k: None)
    supervisor.logger = logging.getLogger("ServiceSupervisorBootstrapBlock")
    supervisor.logger.handlers = []
    supervisor.logger.addHandler(logging.NullHandler())

    supervisor._resolve_bootstrap_handles = lambda: (None, None)
    monkeypatch.setattr(ss, "_BOOTSTRAP_STATE", types.SimpleNamespace(depth=1))
    monkeypatch.setattr(ss, "SelfCodingEngine", lambda *_a, **_k: object())
    monkeypatch.setattr(ss, "CodeDB", lambda *_a, **_k: object())
    monkeypatch.setattr(ss, "MenaceMemoryManager", lambda *_a, **_k: object())

    with pytest.raises(RuntimeError, match="cannot build a pipeline"):
        supervisor.deploy_patch(tmp_path, "desc")


def test_constructor_reuses_inflight_bootstrap(monkeypatch, tmp_path):
    ss.bus = types.SimpleNamespace()
    ss.registry = types.SimpleNamespace()
    ss.data_bot = types.SimpleNamespace()

    bootstrap_pipeline = types.SimpleNamespace(manager="sentinel", initial_manager="sentinel")
    promotion_calls: list[object] = []

    def _promote(manager):
        promotion_calls.append(manager)
        bootstrap_pipeline.manager = manager

    broker = types.SimpleNamespace(resolve=lambda: (bootstrap_pipeline, bootstrap_pipeline.manager))
    bootstrap_state = types.SimpleNamespace(helper_promotion_callbacks=[_promote])

    def _noop_promoter(*_args, **_kwargs):
        promotion_calls.append("fallback")

    prepare_calls: list[dict[str, object]] = []

    def _prepare(**kwargs):
        prepare_calls.append(kwargs)
        return types.SimpleNamespace(), _noop_promoter

    manager = DummyManager()
    monkeypatch.setattr(ss, "prepare_pipeline_for_bootstrap", _prepare)
    monkeypatch.setattr(ss, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(ss, "_BOOTSTRAP_STATE", bootstrap_state)
    monkeypatch.setattr(ss, "_current_bootstrap_context", lambda: types.SimpleNamespace(pipeline=bootstrap_pipeline))
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

    assert prepare_calls == []
    assert promotion_calls == [manager]



def test_dependency_provision_worker_degrades_when_optional(monkeypatch):
    called = {"check": 0}

    class _FailingProvisioner:
        def provision(self):
            raise RuntimeError("boom")

        def provisioning_optional(self):
            return True

    monkeypatch.setattr(ss, "ExternalDependencyProvisioner", _FailingProvisioner)
    monkeypatch.setattr(ss, "DependencyWatchdog", lambda *a, **k: types.SimpleNamespace(check=lambda: called.__setitem__("check", called["check"] + 1)))

    def _interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ss.time, "sleep", _interrupt)

    ss._dependency_provision_worker()
    assert called["check"] == 1


def test_dependency_provision_worker_degraded_unavailable(monkeypatch):
    called = {"check": 0}

    class _Provisioner:
        def provision(self):
            return types.SimpleNamespace(is_unavailable=True, message="no compose", status="unavailable")

        def provisioning_optional(self):
            return False

    monkeypatch.setattr(ss, "ExternalDependencyProvisioner", _Provisioner)
    monkeypatch.setattr(ss, "DependencyWatchdog", lambda *a, **k: types.SimpleNamespace(check=lambda: called.__setitem__("check", called["check"] + 1)))

    def _interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ss.time, "sleep", _interrupt)

    ss._dependency_provision_worker()
    assert called["check"] == 1


def test_dependency_provision_worker_raises_when_not_optional(monkeypatch):
    class _FailingProvisioner:
        def provision(self):
            raise RuntimeError("boom")

        def provisioning_optional(self):
            return False

    monkeypatch.setattr(ss, "ExternalDependencyProvisioner", _FailingProvisioner)

    with pytest.raises(RuntimeError, match="boom"):
        ss._dependency_provision_worker()


def test_self_test_worker_invocation_contract(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeSelfTestService:
        def __init__(self, *args, **kwargs):
            captured["init_kwargs"] = kwargs

        def run_continuous(self, *args, **kwargs):
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(ss, "SelfTestService", _FakeSelfTestService)
    monkeypatch.setenv("SELF_TEST_INTERVAL", "42")

    def _interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ss.time, "sleep", _interrupt)

    ss._self_test_worker(DummyBuilder())

    run_kwargs = captured["run_kwargs"]
    assert "context_builder" in captured["init_kwargs"]
    assert run_kwargs["interval"] == 42.0
    assert "stop_event" in run_kwargs
    assert run_kwargs["stop_event"].is_set()


def test_dependency_watchdog_opens_circuit_and_reports_status(monkeypatch):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.logger = logging.getLogger("ServiceSupervisorWatchdog")
    supervisor.logger.handlers = []
    supervisor.logger.addHandler(logging.NullHandler())
    supervisor.check_interval = 1.0
    supervisor._watchdog_failure_timestamps = []
    supervisor._watchdog_consecutive_failures = 0
    supervisor._watchdog_circuit_state = "closed"
    supervisor._watchdog_opened_at = 0.0
    supervisor._watchdog_next_action_at = 0.0
    supervisor._watchdog_is_degraded = False
    supervisor._watchdog_restart_window_seconds = 300.0
    supervisor._watchdog_circuit_threshold = 2
    supervisor._watchdog_backoff_base_seconds = 2.0
    supervisor._watchdog_backoff_max_seconds = 60.0
    supervisor._watchdog_backoff_jitter_ratio = 0.0
    supervisor._watchdog_probe_interval_seconds = 30.0
    supervisor.processes = {}

    heals: list[str] = []
    supervisor._heal = lambda name: heals.append(name)

    supervisor._handle_dependency_watchdog_exit(100.0)
    supervisor._handle_dependency_watchdog_exit(101.0)

    assert supervisor._watchdog_circuit_state == "open"
    assert supervisor._watchdog_is_degraded is True
    assert heals == ["dependency_watchdog"]

    status = supervisor.status()
    assert status["dependency_watchdog_breaker"]["state"] == "open"
    assert status["dependency_watchdog_breaker"]["degraded"] is True


def test_dependency_watchdog_half_open_probe_and_recovery(monkeypatch):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.logger = logging.getLogger("ServiceSupervisorWatchdogRecovery")
    supervisor.logger.handlers = []
    supervisor.logger.addHandler(logging.NullHandler())
    supervisor.check_interval = 1.0
    supervisor._watchdog_failure_timestamps = [1.0, 2.0]
    supervisor._watchdog_consecutive_failures = 2
    supervisor._watchdog_circuit_state = "open"
    supervisor._watchdog_opened_at = 10.0
    supervisor._watchdog_next_action_at = 15.0
    supervisor._watchdog_is_degraded = True
    supervisor._watchdog_restart_window_seconds = 300.0
    supervisor._watchdog_circuit_threshold = 5
    supervisor._watchdog_backoff_base_seconds = 2.0
    supervisor._watchdog_backoff_max_seconds = 60.0
    supervisor._watchdog_backoff_jitter_ratio = 0.0
    supervisor._watchdog_probe_interval_seconds = 30.0

    heals: list[str] = []
    supervisor._heal = lambda name: heals.append(name)

    supervisor._handle_dependency_watchdog_exit(16.0)
    assert supervisor._watchdog_circuit_state == "half_open"
    assert heals == ["dependency_watchdog"]

    supervisor._handle_dependency_watchdog_alive()
    assert supervisor._watchdog_circuit_state == "closed"
    assert supervisor._watchdog_is_degraded is False
    assert supervisor._watchdog_consecutive_failures == 0
    assert supervisor._watchdog_failure_timestamps == []


def test_dependency_watchdog_backoff_uses_exponential_strategy():
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.check_interval = 1.0
    supervisor._watchdog_consecutive_failures = 3
    supervisor._watchdog_backoff_base_seconds = 2.0
    supervisor._watchdog_backoff_max_seconds = 60.0
    supervisor._watchdog_backoff_jitter_ratio = 0.0

    delay = supervisor._compute_watchdog_backoff()
    assert delay == pytest.approx(8.0)


def test_dependency_watchdog_warning_is_high_signal_and_deduplicated(caplog):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.logger = logging.getLogger("ServiceSupervisorWatchdogSignal")
    supervisor.logger.handlers = []
    supervisor.logger.propagate = True
    supervisor.check_interval = 1.0
    supervisor._watchdog_failure_timestamps = []
    supervisor._watchdog_consecutive_failures = 0
    supervisor._watchdog_circuit_state = "closed"
    supervisor._watchdog_opened_at = 0.0
    supervisor._watchdog_next_action_at = 0.0
    supervisor._watchdog_is_degraded = False
    supervisor._watchdog_restart_window_seconds = 300.0
    supervisor._watchdog_circuit_threshold = 10
    supervisor._watchdog_backoff_base_seconds = 2.0
    supervisor._watchdog_backoff_max_seconds = 60.0
    supervisor._watchdog_backoff_jitter_ratio = 0.0
    supervisor._watchdog_probe_interval_seconds = 30.0
    supervisor._watchdog_last_warning_key = ""

    heals: list[str] = []
    supervisor._heal = lambda name: heals.append(name)

    with caplog.at_level(logging.WARNING):
        supervisor._handle_dependency_watchdog_exit(10.0)
        supervisor._handle_dependency_watchdog_exit(12.0)

    warning_lines = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len([m for m in warning_lines if "crash loop detected" in m]) == 1
    assert all("Traceback" not in m for m in warning_lines)
    assert heals == ["dependency_watchdog", "dependency_watchdog"]


def test_monitor_raises_on_critical_classification(monkeypatch):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.logger = logging.getLogger("ServiceSupervisorCritical")
    supervisor.logger.handlers = []
    supervisor.logger.addHandler(logging.NullHandler())
    supervisor.check_interval = 0.01
    supervisor.processes = {"self_coding_engine": types.SimpleNamespace(is_alive=lambda: False)}
    supervisor.targets = {"self_coding_engine": ss.RegisteredService(target=lambda: None, critical=True)}
    supervisor.healer = types.SimpleNamespace(heal=lambda _name: None)

    monkeypatch.setattr(
        ss,
        "classify_runtime_failure",
        lambda **_kwargs: types.SimpleNamespace(
            category="critical",
            reason="self_coding_engine_failure",
            should_exit=True,
        ),
    )
    monkeypatch.setattr(ss.time, "sleep", lambda _interval: None)

    with pytest.raises(RuntimeError, match="self_coding_engine_failure"):
        supervisor._monitor()


def test_monitor_exits_when_classification_requires_exit(monkeypatch):
    supervisor = object.__new__(ss.ServiceSupervisor)
    supervisor.logger = logging.getLogger("ServiceSupervisorNonCritical")
    supervisor.logger.handlers = []
    supervisor.logger.addHandler(logging.NullHandler())
    supervisor.check_interval = 0.01
    supervisor.processes = {"optional_worker": types.SimpleNamespace(is_alive=lambda: False)}
    supervisor.targets = {"optional_worker": ss.RegisteredService(target=lambda: None, critical=False)}
    heals: list[str] = []
    supervisor.healer = types.SimpleNamespace(heal=lambda name: heals.append(name))

    monkeypatch.setattr(
        ss,
        "classify_runtime_failure",
        lambda **_kwargs: types.SimpleNamespace(
            category="critical",
            reason="optional_worker_failure",
            should_exit=True,
        ),
    )
    sleep_calls = {"count": 0}

    def _sleep_then_stop(_interval):
        sleep_calls["count"] += 1
        if sleep_calls["count"] > 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ss.time, "sleep", _sleep_then_stop)

    with pytest.raises(RuntimeError, match="runtime classification should_exit"):
        supervisor._monitor()

    assert heals == []


def test_runnable_bot_critical_policy_self_improvement_only():
    critical_flags = {entry.name: entry.critical for entry in ss.RUNNABLE_BOT_REGISTRY}

    assert critical_flags["self_learning_service"] is True
    assert critical_flags["self_test_service"] is False


def test_iter_enabled_runnable_bots_filters_env(monkeypatch):
    entries = [
        types.SimpleNamespace(name="always", enabled_if_env=None),
        types.SimpleNamespace(name="gated", enabled_if_env="ENABLE_FOO"),
    ]
    monkeypatch.setattr(ss, "RUNNABLE_BOT_REGISTRY", tuple(entries))
    monkeypatch.delenv("ENABLE_FOO", raising=False)

    selected = ss._iter_enabled_runnable_bots()
    assert [entry.name for entry in selected] == ["always"]

    monkeypatch.setenv("ENABLE_FOO", "1")
    selected = ss._iter_enabled_runnable_bots()
    assert [entry.name for entry in selected] == ["always", "gated"]


def test_main_registers_every_enabled_runnable_bot(monkeypatch):
    entries = [
        types.SimpleNamespace(
            name="always",
            health_endpoint="http://always",
            critical=False,
            liveness_check="process_alive",
            enabled_if_env=None,
        ),
        types.SimpleNamespace(
            name="gated",
            health_endpoint=None,
            critical=True,
            liveness_check="process_alive",
            enabled_if_env="ENABLE_FOO",
        ),
    ]

    class _FakeBootstrapper:
        def bootstrap(self):
            return None

        def readiness_state(self):
            return {"gates": {}}

    registered: list[tuple[str, str | None, bool, str]] = []

    class _FakeSupervisor:
        def __init__(self, context_builder):
            self.context_builder = context_builder

        def register(self, name, target, health_url=None, critical=False, liveness_check="process_alive"):
            registered.append((name, health_url, critical, liveness_check))

        def start_all(self):
            return None

    monkeypatch.setattr(ss, "RUNNABLE_BOT_REGISTRY", tuple(entries))
    monkeypatch.setattr(ss, "EnvironmentBootstrapper", _FakeBootstrapper)
    monkeypatch.setattr(ss, "create_context_builder", lambda: object())
    monkeypatch.setattr(ss, "ServiceSupervisor", _FakeSupervisor)
    monkeypatch.setattr(ss, "_resolve_registry_callable", lambda _entry, _builder: (lambda: None))

    monkeypatch.delenv("ENABLE_FOO", raising=False)
    ss.main()
    assert [name for name, *_ in registered] == ["always"]

    registered.clear()
    monkeypatch.setenv("ENABLE_FOO", "1")
    ss.main()
    assert [name for name, *_ in registered] == ["always", "gated"]
