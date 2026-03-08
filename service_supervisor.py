from __future__ import annotations

"""Process supervisor launching and monitoring Menace services."""

import logging
import logging.handlers
import multiprocessing as mp
import os
import time
import uuid
from importlib import import_module
from functools import partial
from pathlib import Path
from threading import Event
from typing import Callable, Dict, Optional, Tuple

_PACKAGE_CONTEXT = (__package__ or "").strip()
_USE_SCRIPT_IMPORTS = _PACKAGE_CONTEXT == ""
_IMPORT_MODE = "script" if _USE_SCRIPT_IMPORTS else "package"

if _USE_SCRIPT_IMPORTS:
    from bootstrap_timeout_policy import (
        _BOOTSTRAP_TIMEOUT_MINIMUMS,
        derive_bootstrap_timeout_env,
        enforce_bootstrap_timeout_policy,
        guard_bootstrap_wait_env,
    )
else:
    from .bootstrap_timeout_policy import (
        _BOOTSTRAP_TIMEOUT_MINIMUMS,
        derive_bootstrap_timeout_env,
        enforce_bootstrap_timeout_policy,
        guard_bootstrap_wait_env,
    )


def _import_supervisor_module(package_module: str, script_module: str):
    """Import a module in deterministic package/script mode."""
    if _USE_SCRIPT_IMPORTS:
        return import_module(script_module)
    return import_module(package_module, package=_PACKAGE_CONTEXT)


log_record = _import_supervisor_module(".logging_utils", "logging_utils").log_record

def _hydrate_bootstrap_timeout_env() -> dict[str, float]:
    guard_bootstrap_wait_env()
    defaults = derive_bootstrap_timeout_env(
        minimum=_BOOTSTRAP_TIMEOUT_MINIMUMS["MENACE_BOOTSTRAP_WAIT_SECS"]
    )
    for env_var, resolved in defaults.items():
        os.environ.setdefault(env_var, str(resolved))
    return defaults


_BOOTSTRAP_TIMEOUT_DEFAULTS = _hydrate_bootstrap_timeout_env()

BOOTSTRAP_TIMEOUT_ENV = enforce_bootstrap_timeout_policy(logger=logging.getLogger(__name__))

_db_router_module = _import_supervisor_module(".db_router", "db_router")
GLOBAL_ROUTER = _db_router_module.GLOBAL_ROUTER
init_db_router = _db_router_module.init_db_router
resolve_path = _import_supervisor_module(  # type: ignore[assignment]
    ".dynamic_path_router", "dynamic_path_router"
).resolve_path

# Initialise a global DB router with a unique menace_id before importing modules
# that may touch the database. When a router already exists it is reused to
# avoid spawning multiple routers.
MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

_init_unused_bots = _import_supervisor_module(
    ".menace_master", "menace_master"
)._init_unused_bots
MenaceOrchestrator = _import_supervisor_module(
    ".menace_orchestrator", "menace_orchestrator"
).MenaceOrchestrator
MicrotrendService = _import_supervisor_module(
    ".microtrend_service", "microtrend_service"
).MicrotrendService
SelfEvaluationService = _import_supervisor_module(
    ".self_evaluation_service", "self_evaluation_service"
).SelfEvaluationService
learning_main = _import_supervisor_module(
    ".self_learning_service", "self_learning_service"
).main
ModelRankingService = _import_supervisor_module(
    ".cross_model_scheduler", "cross_model_scheduler"
).ModelRankingService
DependencyUpdateService = _import_supervisor_module(
    ".dependency_update_service", "dependency_update_service"
).DependencyUpdateService
_advanced_error_management = _import_supervisor_module(
    ".advanced_error_management", "advanced_error_management"
)
SelfHealingOrchestrator = _advanced_error_management.SelfHealingOrchestrator
AutomatedRollbackManager = _advanced_error_management.AutomatedRollbackManager
KnowledgeGraph = _import_supervisor_module(".knowledge_graph", "knowledge_graph").KnowledgeGraph
ChaosMonitoringService = _import_supervisor_module(
    ".chaos_monitoring_service", "chaos_monitoring_service"
).ChaosMonitoringService
ModelEvaluationService = _import_supervisor_module(
    ".model_evaluation_service", "model_evaluation_service"
).ModelEvaluationService
SecretRotationService = _import_supervisor_module(
    ".secret_rotation_service", "secret_rotation_service"
).SecretRotationService
EnvironmentBootstrapper = _import_supervisor_module(
    ".environment_bootstrap", "environment_bootstrap"
).EnvironmentBootstrapper
ExternalDependencyProvisioner = _import_supervisor_module(
    ".external_dependency_provisioner", "external_dependency_provisioner"
).ExternalDependencyProvisioner
DependencyWatchdog = _import_supervisor_module(
    ".dependency_watchdog", "dependency_watchdog"
).DependencyWatchdog
EnvironmentRestorationService = _import_supervisor_module(
    ".environment_restoration_service", "environment_restoration_service"
).EnvironmentRestorationService
run_startup_checks = _import_supervisor_module(".startup_checks", "startup_checks").run_startup_checks
Autoscaler = _import_supervisor_module(".autoscaler", "autoscaler").Autoscaler
UnifiedUpdateService = _import_supervisor_module(
    ".unified_update_service", "unified_update_service"
).UnifiedUpdateService
SelfTestService = _import_supervisor_module(".self_test_service", "self_test_service").SelfTestService
AutoEscalationManager = _import_supervisor_module(
    ".auto_escalation_manager", "auto_escalation_manager"
).AutoEscalationManager
_self_coding_manager = _import_supervisor_module(
    ".self_coding_manager", "self_coding_manager"
)
PatchApprovalPolicy = _self_coding_manager.PatchApprovalPolicy
_helper_fn = _self_coding_manager._manager_generate_helper_with_builder
internalize_coding_bot = _self_coding_manager.internalize_coding_bot
ErrorDB = _import_supervisor_module(".error_bot", "error_bot").ErrorDB
SelfCodingEngine = _import_supervisor_module(".self_coding_engine", "self_coding_engine").SelfCodingEngine
CodeDB = _import_supervisor_module(".code_database", "code_database").CodeDB
MenaceMemoryManager = _import_supervisor_module(
    ".menace_memory_manager", "menace_memory_manager"
).MenaceMemoryManager
ModelAutomationPipeline = _import_supervisor_module(
    ".model_automation_pipeline", "model_automation_pipeline"
).ModelAutomationPipeline
QuickFixEngine = _import_supervisor_module(".quick_fix_engine", "quick_fix_engine").QuickFixEngine
bus = _import_supervisor_module(".shared_event_bus", "shared_event_bus").event_bus
BotRegistry = _import_supervisor_module(".bot_registry", "bot_registry").BotRegistry
_data_bot_module = _import_supervisor_module(".data_bot", "data_bot")
DataBot = _data_bot_module.DataBot
persist_sc_thresholds = _data_bot_module.persist_sc_thresholds
get_thresholds = _import_supervisor_module(
    ".self_coding_thresholds", "self_coding_thresholds"
).get_thresholds
_coding_bot_interface = _import_supervisor_module(
    ".coding_bot_interface", "coding_bot_interface"
)
_BOOTSTRAP_STATE = _coding_bot_interface._BOOTSTRAP_STATE
_bootstrap_dependency_broker = _coding_bot_interface._bootstrap_dependency_broker
_current_bootstrap_context = _coding_bot_interface._current_bootstrap_context
self_coding_managed = _coding_bot_interface.self_coding_managed
get_orchestrator = _import_supervisor_module(
    ".shared_evolution_orchestrator", "shared_evolution_orchestrator"
).get_orchestrator
if _USE_SCRIPT_IMPORTS:
    from context_builder_util import create_context_builder  # noqa: E402
    from vector_service.context_builder import ContextBuilder  # noqa: E402
else:
    from .context_builder_util import create_context_builder  # noqa: E402
    from .vector_service.context_builder import ContextBuilder  # noqa: E402

try:  # optional dependency
    import psutil  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional
    psutil = None  # type: ignore

logging.getLogger(__name__).info(
    "service_supervisor import bootstrap mode selected: %s",
    _IMPORT_MODE,
)

# ``bus`` is the shared :class:`UnifiedEventBus` instance so all services
# exchange events via a common channel.  Creating separate buses would isolate
# degradation notifications and break global coordination.
registry = BotRegistry(event_bus=bus)
data_bot = DataBot(event_bus=bus)


# ---------------------------------------------------------------------------
# Worker functions for individual services
# ---------------------------------------------------------------------------

def _parse_map(value: str) -> dict[str, str]:
    pairs = [p.strip() for p in value.split(',') if p.strip()]
    result: dict[str, str] = {}
    for pair in pairs:
        if '=' in pair:
            k, v = pair.split('=', 1)
            result[k.strip()] = v.strip()
    return result


def _orchestrator_worker(context_builder: ContextBuilder) -> None:
    """Run the main Menace orchestration loop."""
    logger = logging.getLogger("orchestrator_worker")
    _init_unused_bots()
    models = os.environ.get("MODELS", "demo").split(",")
    sleep_seconds = float(os.environ.get("SLEEP_SECONDS", "0"))
    orch = MenaceOrchestrator(context_builder=context_builder)
    orch.create_oversight("root", "L1")
    if sleep_seconds > 0:
        orch.start_scheduled_jobs()
    try:
        while True:
            results = orch.run_cycle(models)
            for model, res in results.items():
                print(f"{model}: {res}")
            if sleep_seconds <= 0:
                break
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        logger.info("orchestrator worker interrupted")
    finally:
        if sleep_seconds > 0:
            orch.stop_scheduled_jobs()


def _microtrend_worker() -> None:
    """Continuously run the microtrend service."""
    logger = logging.getLogger("microtrend_worker")
    service = MicrotrendService()
    stop = Event()
    service.run_continuous(
        interval=float(os.getenv("MICROTREND_INTERVAL", "3600")),
        stop_event=stop,
    )
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("microtrend worker interrupted")
        stop.set()


def _self_eval_worker() -> None:
    """Run the self-evaluation service combining trends and cloning."""
    logger = logging.getLogger("self_eval_worker")
    service = SelfEvaluationService()
    stop = Event()
    interval = float(os.getenv("SELF_EVAL_INTERVAL", "3600"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("self evaluation worker interrupted")
        stop.set()


def _learning_worker() -> None:
    """Run the self-learning coordinator."""
    learning_main(stop_event=Event())


def _ranking_worker() -> None:
    """Periodically rank models and redeploy the best."""
    logger = logging.getLogger("ranking_worker")
    service = ModelRankingService()
    stop = Event()
    interval = float(os.getenv("MODEL_RANK_INTERVAL", "86400"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("ranking worker interrupted")
        stop.set()


def _dep_update_worker() -> None:
    """Periodically update dependencies and redeploy."""
    logger = logging.getLogger("dependency_update_worker")
    service = DependencyUpdateService()
    stop = Event()
    interval = float(os.getenv("DEP_UPDATE_INTERVAL", "86400"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("dependency update worker interrupted")
        stop.set()


def _chaos_worker(context_builder: ContextBuilder) -> None:
    """Continuously inject faults and rollback on failure."""
    logger = logging.getLogger("chaos_worker")
    service = ChaosMonitoringService(context_builder=context_builder)
    stop = Event()
    interval = float(os.getenv("CHAOS_INTERVAL", "300"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("chaos worker interrupted")
        stop.set()


def _eval_worker() -> None:
    """Periodic self-hosted model evaluation."""
    logger = logging.getLogger("evaluation_worker")
    service = ModelEvaluationService()
    stop = Event()
    interval = float(os.getenv("EVAL_INTERVAL", "43200"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("evaluation worker interrupted")
        stop.set()


def _debug_worker(context_builder: ContextBuilder) -> None:
    """Continuously run telemetry-driven debugging."""
    from .debug_loop_service import DebugLoopService
    logger = logging.getLogger("debug_worker")
    service = DebugLoopService(context_builder=context_builder)
    stop = Event()
    interval = float(os.getenv("DEBUG_INTERVAL", "300"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("debug worker interrupted")
        stop.set()


def _dependency_provision_worker() -> None:
    """Provision external dependencies and monitor them."""
    logger = logging.getLogger("dependency_provision_worker")
    ExternalDependencyProvisioner().provision()
    interval = float(os.getenv("WATCHDOG_INTERVAL", "60"))
    endpoints = _parse_map(os.getenv("DEPENDENCY_ENDPOINTS", ""))
    backups = _parse_map(os.getenv("DEPENDENCY_BACKUPS", ""))
    watchdog = DependencyWatchdog(endpoints, backups)
    try:
        while True:
            watchdog.check()
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("dependency provision worker interrupted")


def _dependency_monitor_worker() -> None:
    """Periodically verify critical dependencies and config."""
    logger = logging.getLogger("dependency_monitor_worker")
    interval = float(os.getenv("DEPENDENCY_MONITOR_INTERVAL", "3600"))
    try:
        while True:
            try:
                run_startup_checks()
            except Exception as exc:
                logging.getLogger("dependency_monitor").error(
                    "dependency check failed: %s", exc
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("dependency monitor worker interrupted")


def _env_restore_worker() -> None:
    """Continuously restore the environment using the bootstrapper."""
    logger = logging.getLogger("env_restore_worker")
    service = EnvironmentRestorationService()
    stop = Event()
    interval = float(os.getenv("ENV_RESTORE_INTERVAL", "3600"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("environment restoration worker interrupted")
        stop.set()


def _update_worker() -> None:
    """Run unified dependency updates and redeploy."""
    logger = logging.getLogger("update_worker")
    svc = UnifiedUpdateService()
    stop = Event()
    interval = float(os.getenv("UPDATE_INTERVAL", "86400"))
    svc.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("update worker interrupted")
        stop.set()


def _self_test_worker(builder: ContextBuilder) -> None:
    """Execute the self test suite periodically."""
    logger = logging.getLogger("self_test_worker")
    svc = SelfTestService(context_builder=builder)
    stop = Event()
    interval = float(os.getenv("SELF_TEST_INTERVAL", "86400"))
    svc.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("self test worker interrupted")
        stop.set()


def _autoscale_worker() -> None:
    """Adjust resources based on system load."""
    logger = logging.getLogger("autoscale_worker")
    auto = Autoscaler()
    interval = float(os.getenv("AUTOSCALE_INTERVAL", "0"))
    if interval <= 0:
        interval = 60.0
    try:
        while True:
            cpu = psutil.cpu_percent() / 100.0 if psutil else 0.0
            mem = (
                psutil.virtual_memory().percent / 100.0
                if psutil
                else 0.0
            )
            try:
                auto.scale({"cpu": cpu, "memory": mem})
            except Exception as exc:  # pragma: no cover - best effort
                logging.getLogger("autoscale_worker").error(
                    "autoscale failed: %s", exc
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("autoscale worker interrupted")


def _secret_rotation_worker() -> None:
    """Periodically rotate configured secrets."""
    logger = logging.getLogger("secret_rotation_worker")
    service = SecretRotationService()
    stop = Event()
    interval = float(os.getenv("SECRET_ROTATION_INTERVAL", "86400"))
    service.run_continuous(interval=interval, stop_event=stop)
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("secret rotation worker interrupted")
        stop.set()


# ---------------------------------------------------------------------------
@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class ServiceSupervisor:
    """Supervisor managing Menace background processes."""

    def __init__(
        self,
        check_interval: float = 5.0,
        *,
        context_builder: ContextBuilder,
        log_path: str = "supervisor.log",
        restart_log: str = "restart.log",
        dependency_broker: object | None = None,
        bootstrap_context: object | None = None,
        pipeline: object | None = None,
        pipeline_promoter: Callable[[object], None] | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=10**6, backupCount=3)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.check_interval = check_interval
        self.restart_log = restart_log
        self.targets: Dict[str, Tuple[Callable[[], None], Optional[str]]] = {}
        self.processes: Dict[str, mp.Process] = {}
        self._pipeline_promoter: Callable[[object], None] | None = pipeline_promoter
        self._bootstrap_dependency_broker = (
            dependency_broker if dependency_broker is not None else _bootstrap_dependency_broker()
        )
        self._bootstrap_context = bootstrap_context
        # Use the self healing orchestrator for restart logic
        self.healer = SelfHealingOrchestrator(KnowledgeGraph())
        self.healer.heal = self._heal  # type: ignore[assignment]
        self.rollback_mgr = AutomatedRollbackManager()
        self.approval_policy = PatchApprovalPolicy(
            rollback_mgr=self.rollback_mgr, bot_name="menace"
        )
        self.context_builder = context_builder
        self.context_builder.refresh_db_weights()
        self.auto_mgr = AutoEscalationManager(context_builder=self.context_builder)
        engine = SelfCodingEngine(
            CodeDB(), MenaceMemoryManager(), context_builder=self.context_builder
        )
        bootstrap_pipeline, bootstrap_promoter = self._resolve_bootstrap_handles()
        if pipeline is None:
            pipeline = bootstrap_pipeline
        if self._pipeline_promoter is None:
            self._pipeline_promoter = bootstrap_promoter
        if pipeline is None:
            bootstrap_depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
            if bootstrap_depth > 0 or bootstrap_pipeline is not None:
                raise RuntimeError(
                    "ServiceSupervisor cannot create a pipeline while bootstrap is active"
                )
            raise RuntimeError(
                "ServiceSupervisor requires an existing ModelAutomationPipeline"
            )
        evolution_orchestrator = get_orchestrator(
            self.__class__.__name__, data_bot, engine
        )
        _th = get_thresholds(self.__class__.__name__)
        persist_sc_thresholds(
            self.__class__.__name__,
            roi_drop=_th.roi_drop,
            error_increase=_th.error_increase,
            test_failure_increase=_th.test_failure_increase,
            event_bus=bus,
        )
        manager = internalize_coding_bot(
            self.__class__.__name__,
            engine,
            pipeline,
            data_bot=data_bot,
            bot_registry=registry,
            approval_policy=self.approval_policy,
            event_bus=bus,
            roi_threshold=_th.roi_drop,
            error_threshold=_th.error_increase,
            test_failure_threshold=_th.test_failure_increase,
            evolution_orchestrator=evolution_orchestrator,
        )
        promoter = self._pipeline_promoter
        if promoter is not None:
            promoter(manager)
            self._pipeline_promoter = None
        manager.context_builder = self.context_builder
        self.manager = manager
        self.error_db = ErrorDB()
        self.fix_engine = QuickFixEngine(
            self.error_db,
            manager,
            graph=self.healer.graph,
            context_builder=self.context_builder,
            helper_fn=_helper_fn,
        )
        manager.quick_fix = self.fix_engine
        self.evolution_orchestrator = evolution_orchestrator

    def _resolve_bootstrap_handles(
        self,
    ) -> tuple[object | None, Callable[[object], None] | None]:
        """Return any active bootstrap pipeline/promoter for reuse."""

        pipeline_candidate = None
        promoter: Callable[[object], None] | None = None

        broker: object | None = getattr(self, "_bootstrap_dependency_broker", None)
        if callable(broker) and not hasattr(broker, "resolve"):
            try:
                broker = broker()
            except Exception:
                broker = None
        if broker is None:
            try:
                broker = _bootstrap_dependency_broker()
            except Exception:
                broker = None
            self._bootstrap_dependency_broker = broker
        if broker is not None:
            try:
                pipeline_candidate, _sentinel = broker.resolve()
            except Exception:
                pipeline_candidate = None

        if getattr(self, "_bootstrap_context", None) is None:
            try:
                self._bootstrap_context = _current_bootstrap_context()
            except Exception:
                self._bootstrap_context = None

        if pipeline_candidate is None and getattr(self, "_bootstrap_context", None) is not None:
            pipeline_candidate = getattr(self._bootstrap_context, "pipeline", None)

        callbacks = getattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks", None)
        if callbacks:
            promoter = callbacks[-1]

        return pipeline_candidate, promoter

    def _record_failure(self, etype: str) -> None:
        """Record supervisor issues to the knowledge graph."""
        try:
            self.healer.graph.add_telemetry_event("service_supervisor", etype)
        except Exception as exc:  # pragma: no cover - graph failures
            self.logger.error("telemetry logging failed: %s", exc)

    # ------------------------------------------------------------------
    def deploy_patch(self, path: Path, description: str) -> None:
        """Apply a patch using the approval policy."""
        try:
            from .self_coding_engine import SelfCodingEngine
            from .code_database import CodeDB
            from .menace_memory_manager import MenaceMemoryManager
            from .model_automation_pipeline import ModelAutomationPipeline

            engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=self.context_builder
            )
            pipeline, promote_pipeline = self._resolve_bootstrap_handles()
            if pipeline is None and hasattr(self, "manager"):
                try:
                    pipeline = getattr(self.manager, "pipeline", None)
                except Exception:
                    pipeline = None
            if pipeline is None:
                bootstrap_depth = getattr(_BOOTSTRAP_STATE, "depth", 0)
                if bootstrap_depth > 0:
                    raise RuntimeError(
                        "deploy_patch cannot build a pipeline while bootstrap is active"
                    )
                raise RuntimeError(
                    "deploy_patch requires an existing ModelAutomationPipeline"
                )
            evolution_orchestrator = get_orchestrator(
                self.__class__.__name__, data_bot, engine
            )
            _th = get_thresholds(self.__class__.__name__)
            persist_sc_thresholds(
                self.__class__.__name__,
                roi_drop=_th.roi_drop,
                error_increase=_th.error_increase,
                test_failure_increase=_th.test_failure_increase,
                event_bus=bus,
            )
            manager = internalize_coding_bot(
                self.__class__.__name__,
                engine,
                pipeline,
                data_bot=data_bot,
                bot_registry=registry,
                approval_policy=self.approval_policy,
                event_bus=bus,
                roi_threshold=_th.roi_drop,
                error_threshold=_th.error_increase,
                test_failure_threshold=_th.test_failure_increase,
                evolution_orchestrator=evolution_orchestrator,
            )
            if promote_pipeline is not None:
                promote_pipeline(manager)
            manager.context_builder = self.context_builder
            outcome = manager.auto_run_patch(path, description)
            summary = outcome.get("summary") if outcome else None
            if summary is None:
                raise RuntimeError("post validation summary unavailable")
            if "self_tests" not in summary:
                raise RuntimeError("self test summary unavailable")
            failed_tests = int(summary["self_tests"].get("failed", 0))
            if failed_tests:
                raise RuntimeError(f"self tests failed ({failed_tests})")
            added_modules = getattr(engine, "last_added_modules", None)
            if not added_modules:
                added_modules = getattr(engine, "added_modules", None)
            if added_modules:
                try:
                    from sandbox_runner import try_integrate_into_workflows
                    try_integrate_into_workflows(
                        added_modules, context_builder=self.context_builder
                    )
                except Exception as wf_exc:
                    self.logger.warning(
                        "workflow integration failed after patch: %s", wf_exc
                    )
        except Exception as exc:
            self.logger.error("patch deployment failed: %s", exc)
            if self.rollback_mgr:
                try:
                    self.rollback_mgr.auto_rollback("latest", [])
                except Exception as rb_exc:
                    self.logger.warning("auto rollback failed: %s", rb_exc)
                    self._record_failure("rollback_failure")

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        target: Callable[[], None],
        health_url: str | None = None,
    ) -> None:
        self.targets[name] = (target, health_url)

    # ------------------------------------------------------------------
    def _start(self, name: str) -> None:
        target, _ = self.targets[name]
        proc = mp.Process(target=target, name=name, daemon=True)
        proc.start()
        self.processes[name] = proc
        self.logger.info("started %s [pid=%s]", name, proc.pid)
        try:
            with open(self.restart_log, "a", encoding="utf-8") as f:
                f.write(f"{time.time()}: started {name} pid={proc.pid}\n")
        except Exception as exc:
            self.logger.warning("failed writing restart log: %s", exc)
            self._record_failure("restart_log_failure")

    # ------------------------------------------------------------------
    def _heal(self, bot: str, patch_id: str | None = None) -> None:
        # Try quick fix then restart the given bot process
        try:
            self.fix_engine.run(bot)
        except Exception as exc:
            self.logger.error("quick fix failed: %s", exc)
        if bot in self.processes:
            self.logger.warning("restarting %s", bot)
            self._start(bot)
            try:
                with open(self.restart_log, "a", encoding="utf-8") as f:
                    f.write(f"{time.time()}: restarted {bot}\n")
            except Exception as exc:
                self.logger.warning("failed writing restart log: %s", exc)
                self._record_failure("restart_log_failure")
            try:
                self.auto_mgr.handle(f"{bot} restarted")
            except Exception as exc:
                self.logger.warning("auto escalation failed: %s", exc)
                self._record_failure("escalation_failure")

    # ------------------------------------------------------------------
    def start_all(self) -> None:
        for name in self.targets:
            self._start(name)
        self._monitor()

    # ------------------------------------------------------------------
    def _monitor(self) -> None:
        while True:
            time.sleep(self.check_interval)
            for name, proc in list(self.processes.items()):
                if not proc.is_alive():
                    self.healer.heal(name)
                    continue
                url = self.targets[name][1]
                if url:
                    try:
                        self.healer.probe_and_heal(name, url)
                    except Exception as exc:
                        self.logger.warning("health probe failed for %s: %s", name, exc)
                        self._record_failure("probe_failure")


# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point starting the service supervisor."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "bootstrap timeout configuration",
        extra={
            "event": "bootstrap-timeouts",
            **BOOTSTRAP_TIMEOUT_ENV,
        },
    )
    bootstrapper = EnvironmentBootstrapper()
    bootstrapper.bootstrap()
    readiness_snapshot = bootstrapper.readiness_state()
    logging.getLogger(__name__).info(
        "bootstrap readiness summary",
        extra=log_record(
            event="bootstrap-readiness-summary",
            gates=readiness_snapshot.get("gates"),
            readiness_tokens=readiness_snapshot,
        ),
    )
    builder = create_context_builder()
    sup = ServiceSupervisor(context_builder=builder)
    sup.register("orchestrator", partial(_orchestrator_worker, builder))
    sup.register("microtrend_service", _microtrend_worker)
    sup.register("self_evaluation_service", _self_eval_worker)
    sup.register("self_learning_service", _learning_worker)
    sup.register("model_ranking_service", _ranking_worker)
    sup.register("dependency_update_service", _dep_update_worker)
    sup.register("chaos_monitoring_service", partial(_chaos_worker, builder))
    sup.register("model_evaluation_service", _eval_worker)
    sup.register("debug_loop_service", partial(_debug_worker, builder))
    sup.register("dependency_watchdog", _dependency_provision_worker)
    sup.register("dependency_monitor", _dependency_monitor_worker)
    sup.register("environment_restoration", _env_restore_worker)
    sup.register("unified_update_service", _update_worker)
    sup.register("self_test_service", partial(_self_test_worker, builder))
    if os.getenv("ENABLE_AUTOSCALER") == "1":
        sup.register("autoscaler", _autoscale_worker)
    if os.getenv("AUTO_ROTATE_SECRETS") == "1":
        sup.register("secret_rotation_service", _secret_rotation_worker)
    sup.start_all()


__all__ = ["ServiceSupervisor", "main"]
