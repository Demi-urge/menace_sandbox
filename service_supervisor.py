from __future__ import annotations

"""Process supervisor launching and monitoring Menace services."""

import logging
import logging.handlers
import multiprocessing as mp
import os
import time
import uuid
from functools import partial
from pathlib import Path
from threading import Event
from typing import Callable, Dict, Optional, Tuple

from .db_router import GLOBAL_ROUTER, init_db_router
try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

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

from .menace_master import _init_unused_bots  # noqa: E402
from .menace_orchestrator import MenaceOrchestrator  # noqa: E402
from .microtrend_service import MicrotrendService  # noqa: E402
from .self_evaluation_service import SelfEvaluationService  # noqa: E402
from .self_learning_service import main as learning_main  # noqa: E402
from .cross_model_scheduler import ModelRankingService  # noqa: E402
from .dependency_update_service import DependencyUpdateService  # noqa: E402
from .advanced_error_management import SelfHealingOrchestrator  # noqa: E402
from .knowledge_graph import KnowledgeGraph  # noqa: E402
from .chaos_monitoring_service import ChaosMonitoringService  # noqa: E402
from .model_evaluation_service import ModelEvaluationService  # noqa: E402
from .secret_rotation_service import SecretRotationService  # noqa: E402
from .environment_bootstrap import EnvironmentBootstrapper  # noqa: E402
from .external_dependency_provisioner import ExternalDependencyProvisioner  # noqa: E402
from .dependency_watchdog import DependencyWatchdog  # noqa: E402
from .environment_restoration_service import EnvironmentRestorationService  # noqa: E402
from .startup_checks import run_startup_checks  # noqa: E402
from .autoscaler import Autoscaler  # noqa: E402
from .unified_update_service import UnifiedUpdateService  # noqa: E402
from .self_test_service import SelfTestService  # noqa: E402
from .auto_escalation_manager import AutoEscalationManager  # noqa: E402
from .self_coding_manager import PatchApprovalPolicy, SelfCodingManager  # noqa: E402
from .advanced_error_management import AutomatedRollbackManager  # noqa: E402
from .error_bot import ErrorDB  # noqa: E402
from .self_coding_engine import SelfCodingEngine  # noqa: E402
from .code_database import CodeDB  # noqa: E402
from .menace_memory_manager import MenaceMemoryManager  # noqa: E402
from .model_automation_pipeline import ModelAutomationPipeline  # noqa: E402
from .quick_fix_engine import QuickFixEngine  # noqa: E402
from vector_service.context_builder import ContextBuilder  # noqa: E402
from context_builder_util import create_context_builder  # noqa: E402
from .unified_event_bus import UnifiedEventBus  # noqa: E402
from .bot_registry import BotRegistry  # noqa: E402
from .data_bot import DataBot  # noqa: E402

try:  # optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore


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
class ServiceSupervisor:
    """Supervisor managing Menace background processes."""

    def __init__(
        self,
        check_interval: float = 5.0,
        *,
        context_builder: ContextBuilder,
        log_path: str = "supervisor.log",
        restart_log: str = "restart.log",
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
        bus = UnifiedEventBus()
        registry = BotRegistry(event_bus=bus)
        data_bot = DataBot(event_bus=bus)
        pipeline = ModelAutomationPipeline(
            context_builder=self.context_builder,
            event_bus=bus,
            bot_registry=registry,
        )
        manager = SelfCodingManager(
            engine,
            pipeline,
            bot_name="menace",
            approval_policy=self.approval_policy,
            data_bot=data_bot,
            bot_registry=registry,
            event_bus=bus,
        )
        manager.context_builder = self.context_builder
        self.error_db = ErrorDB()
        self.fix_engine = QuickFixEngine(
            self.error_db,
            manager,
            graph=self.healer.graph,
            context_builder=self.context_builder,
        )

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
            from .data_bot import DataBot

            engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=self.context_builder
            )
            bus = UnifiedEventBus()
            registry = BotRegistry(event_bus=bus)
            data_bot = DataBot(event_bus=bus)
            pipeline = ModelAutomationPipeline(
                context_builder=self.context_builder,
                event_bus=bus,
                bot_registry=registry,
            )
            manager = SelfCodingManager(
                engine,
                pipeline,
                bot_name="menace",
                approval_policy=self.approval_policy,
                data_bot=data_bot,
                bot_registry=registry,
                event_bus=bus,
            )
            manager.context_builder = self.context_builder
            manager.run_patch(path, description)
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
    EnvironmentBootstrapper().bootstrap()
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
