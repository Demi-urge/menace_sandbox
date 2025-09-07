"""Command line interface running the Menace orchestration loop.

This module contains no graphical components and executes all automation
headlessly.  The optional :mod:`menace_gui` is *not* required for
``menace_master`` to operate and can be ignored when running autonomous
workflows.
"""

from __future__ import annotations

import os
import time
import threading
from typing import Iterable
import argparse
import importlib
import platform
import shutil
import subprocess
import uuid

try:  # optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

from pathlib import Path
import sys
import logging
from logging_utils import log_record
from dynamic_path_router import resolve_path

# Logger for this module
logger = logging.getLogger(__name__)

# Ensure the repository root is available on ``sys.path`` when executing
# directly from a working tree.
ROOT = resolve_path(".")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from menace.db_router import init_db_router  # noqa: E402
except Exception:  # pragma: no cover - fallback for tests
    def init_db_router(*args, **kwargs):  # type: ignore[override]
        return None

MENACE_ID = os.getenv("MENACE_ID", uuid.uuid4().hex)
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

from menace.unified_config_store import UnifiedConfigStore  # noqa: E402
from menace.dependency_self_check import self_check  # noqa: E402

from menace.menace_orchestrator import MenaceOrchestrator  # noqa: E402
from vector_service.context_builder import ContextBuilder  # noqa: E402
from menace.self_coding_manager import PatchApprovalPolicy, SelfCodingManager  # noqa: E402
from menace.advanced_error_management import AutomatedRollbackManager  # noqa: E402
from menace.environment_bootstrap import EnvironmentBootstrapper  # noqa: E402
from menace.auto_env_setup import ensure_env, interactive_setup  # noqa: E402
from menace.auto_resource_setup import ensure_proxies, ensure_accounts  # noqa: E402
from menace.external_dependency_provisioner import ExternalDependencyProvisioner  # noqa: E402
from menace.unified_event_bus import UnifiedEventBus  # noqa: E402
from menace.retry_utils import retry  # noqa: E402
from menace.disaster_recovery import DisasterRecovery  # noqa: E402
try:
    import sandbox_runner  # noqa: E402
except Exception:  # pragma: no cover - fallback for tests
    import types  # noqa: E402
    import sys  # noqa: E402

    def _fallback_run_sandbox(*args, **kwargs):  # type: ignore[override]
        return None

    sandbox_runner = types.SimpleNamespace(_run_sandbox=_fallback_run_sandbox)  # type: ignore
    sys.modules["sandbox_runner"] = sandbox_runner
else:
    if not hasattr(sandbox_runner, "_run_sandbox"):
        def _fallback_run_sandbox(*args, **kwargs):  # type: ignore[override]
            return None

        sandbox_runner._run_sandbox = _fallback_run_sandbox  # type: ignore
from menace.bot_development_bot import BotDevelopmentBot  # noqa: E402
from menace.bot_testing_bot import BotTestingBot  # noqa: E402
from menace.chatgpt_enhancement_bot import ChatGPTEnhancementBot  # noqa: E402
from menace.chatgpt_prediction_bot import ChatGPTPredictionBot  # noqa: E402
from menace.chatgpt_research_bot import ChatGPTResearchBot  # noqa: E402
from menace.competitive_intelligence_bot import CompetitiveIntelligenceBot  # noqa: E402
from menace.contrarian_model_bot import ContrarianModelBot  # noqa: E402
from menace.conversation_manager_bot import ConversationManagerBot  # noqa: E402
from menace.database_steward_bot import DatabaseStewardBot  # noqa: E402
from menace.deployment_bot import DeploymentBot  # noqa: E402
from menace.enhancement_bot import EnhancementBot  # noqa: E402
from menace.error_bot import ErrorBot  # noqa: E402
from menace.ga_prediction_bot import GAPredictionBot  # noqa: E402
from menace.genetic_algorithm_bot import GeneticAlgorithmBot  # noqa: E402
from menace.ipo_bot import IPOBot  # noqa: E402
from menace.implementation_optimiser_bot import ImplementationOptimiserBot  # noqa: E402
from menace.mirror_bot import MirrorBot  # noqa: E402
from menace.niche_saturation_bot import NicheSaturationBot  # noqa: E402
from menace.market_manipulation_bot import MarketManipulationBot  # noqa: E402
from menace.passive_discovery_bot import PassiveDiscoveryBot  # noqa: E402
from menace.preliminary_research_bot import PreliminaryResearchBot  # noqa: E402
from menace.report_generation_bot import ReportGenerationBot  # noqa: E402
from menace.resource_allocation_bot import ResourceAllocationBot  # noqa: E402
from menace.resources_bot import ResourcesBot  # noqa: E402
from menace.scalability_assessment_bot import ScalabilityAssessmentBot  # noqa: E402
from menace.strategy_prediction_bot import StrategyPredictionBot  # noqa: E402
from menace.structural_evolution_bot import StructuralEvolutionBot  # noqa: E402
from menace.text_research_bot import TextResearchBot  # noqa: E402
from menace.video_research_bot import VideoResearchBot  # noqa: E402
from menace.ai_counter_bot import AICounterBot  # noqa: E402
from menace.dynamic_resource_allocator_bot import DynamicResourceAllocator  # noqa: E402
from menace.diagnostic_manager import DiagnosticManager  # noqa: E402
from menace.idea_search_bot import KeywordBank  # noqa: E402
from menace.newsreader_bot import NewsDB  # noqa: E402
from menace.chatgpt_idea_bot import ChatGPTClient  # noqa: E402
try:
    from menace.shared_knowledge_module import LOCAL_KNOWLEDGE_MODULE  # noqa: E402
except Exception:  # pragma: no cover - fallback for tests
    from types import SimpleNamespace

    LOCAL_KNOWLEDGE_MODULE = SimpleNamespace(memory=None)  # type: ignore
from menace.self_learning_service import main as learning_service_main  # noqa: E402
from menace.self_service_override import SelfServiceOverride  # noqa: E402
from menace.resource_allocation_optimizer import ROIDB  # noqa: E402
from menace.data_bot import MetricsDB  # noqa: E402


def _parse_map(value: str) -> dict[str, str]:
    pairs = [p.strip() for p in value.split(",") if p.strip()]
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Return command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=os.getenv("MENACE_ENV_FILE", ".env"),
        help="Path to the environment file",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="Override environment variables",
    )
    parser.add_argument(
        "--interactive-setup",
        action="store_true",
        default=False,
        help="Deprecated: previously prompted for missing API keys",
    )
    parser.add_argument(
        "--defaults-file",
        default=os.getenv("MENACE_DEFAULTS_FILE", ""),
        help="File containing default API key values",
    )
    parser.add_argument(
        "--run-cycles",
        type=int,
        default=None,
        help="Stop after N orchestrator cycles",
    )
    parser.add_argument(
        "--run-until",
        type=float,
        default=None,
        help="Unix timestamp when execution should stop",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        default=False,
        help="Run Menace in a sandboxed temporary clone",
    )
    parser.add_argument(
        "--sandbox-data-dir",
        default=os.getenv("SANDBOX_DATA_DIR", ""),
        help="Directory persisting sandbox state",
    )
    parser.add_argument(
        "--first-run-flag",
        default=os.getenv("MENACE_FIRST_RUN_FILE", ".menace_first_run"),
        help="Path to the first run sentinel file",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        default=False,
        help="Skip automatic dependency installation",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _start_dependency_watchdog(
    event_bus: UnifiedEventBus | None = None,
) -> threading.Thread | None:
    """Provision dependencies automatically and launch a watchdog."""
    attempts = int(os.getenv("PROVISION_ATTEMPTS", "3"))
    delay = float(os.getenv("PROVISION_RETRY_DELAY", "5"))

    @retry(Exception, attempts=attempts, delay=delay)
    def _provision() -> None:
        ExternalDependencyProvisioner().provision()

    endpoints = _parse_map(os.getenv("DEPENDENCY_ENDPOINTS", ""))
    backups = _parse_map(os.getenv("DEPENDENCY_BACKUPS", ""))
    need_provision = not endpoints

    if requests is None:
        try:
            from menace.system_provisioner import SystemProvisioner

            SystemProvisioner(
                packages=["python3-requests", "requests"]
            ).ensure_packages()
            globals()["requests"] = importlib.import_module("requests")
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("dependency watchdog disabled: %s", exc)
            return None

    if requests is not None and endpoints:
        for url in endpoints.values():
            try:
                r = requests.get(url, timeout=3)
                if r.status_code != 200:
                    need_provision = True
                    break
            except Exception:
                need_provision = True
                break

    if need_provision:
        try:
            _provision()
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("dependency provisioning failed: %s", exc)
            if event_bus:
                try:
                    event_bus.publish(
                        "dependency:provision_failed", {"error": str(exc)}
                    )
                except Exception:
                    logger.exception("failed publishing provisioning event")

    if requests is None or not endpoints:
        return None

    from menace.dependency_watchdog import DependencyWatchdog

    watchdog = DependencyWatchdog(endpoints, backups)
    interval = float(os.getenv("WATCHDOG_INTERVAL", "60"))

    def _loop() -> None:
        while True:
            watchdog.check()
            time.sleep(interval)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _install_user_systemd() -> None:
    service_file = resolve_path("systemd/menace.service")
    target = Path.home() / ".config" / "systemd" / "user" / "menace.service"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(service_file, target)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "menace"], check=True)


def _install_task_scheduler() -> None:
    exe = sys.executable
    script = resolve_path("service_supervisor.py")
    cmd = [
        "schtasks",
        "/Create",
        "/TN",
        "menace",
        "/TR",
        f'"{exe}" "{script}"',
        "/SC",
        "ONLOGON",
        "/RL",
        "HIGHEST",
        "/F",
    ]
    subprocess.run(cmd, check=True)


def _auto_service_setup() -> None:
    """Attempt automatic service installation when running with privileges."""
    if os.getenv("AUTO_SERVICE_SETUP", "1").lower() in {"0", "false", "no"}:
        return
    sys_platform = platform.system()
    is_root = False
    if sys_platform == "Windows":
        try:
            import ctypes  # type: ignore

            is_root = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:  # pragma: no cover - best effort
            is_root = False
    else:
        is_root = hasattr(os, "geteuid") and os.geteuid() == 0

    if is_root:
        try:
            from menace.service_installer import _install_windows, _install_systemd

            if sys_platform == "Windows":
                _install_windows()
            else:
                _install_systemd()
            return
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("service setup failed: %s", exc)
            return

    try:
        if sys_platform == "Windows":
            _install_task_scheduler()
        elif sys_platform in {"Linux", "Darwin"}:
            _install_user_systemd()
        else:
            logger.info(
                "Run 'python service_installer.py' to install the Menace service."
            )
            return
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("service setup failed: %s", exc)


def _init_unused_bots() -> None:
    """Instantiate helper bots that are not used elsewhere."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        from vector_service.context_builder import ContextBuilder  # type: ignore

        builder = ContextBuilder(
            bot_db="bots.db",
            code_db="code.db",
            error_db="errors.db",
            workflow_db="workflows.db",
        )
        builder.refresh_db_weights()
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("failed to initialize ContextBuilder", exc_info=True)

        class _DummyBuilder:
            def refresh_db_weights(self):
                pass

            def build(self, query: str, **_: object) -> str:
                return ""

        builder = _DummyBuilder()

    client = ChatGPTClient(
        api_key, gpt_memory=LOCAL_KNOWLEDGE_MODULE.memory, context_builder=builder
    )

    from functools import partial

    def _wrap(cls, *args, **kwargs):
        fn = partial(cls, *args, **kwargs)
        fn.__name__ = getattr(cls, "__name__", repr(cls))
        return fn

    bot_classes = [
        _wrap(BotDevelopmentBot, context_builder=builder),
        BotTestingBot,
        _wrap(ChatGPTEnhancementBot, client),
        _wrap(ChatGPTPredictionBot, client=client, context_builder=builder),
        _wrap(ChatGPTResearchBot, client),
        CompetitiveIntelligenceBot,
        _wrap(
            ContrarianModelBot,
            allocator=ResourceAllocationBot(context_builder=builder),
        ),
        _wrap(ConversationManagerBot, client),
        DatabaseStewardBot,
        DeploymentBot,
        _wrap(EnhancementBot, context_builder=builder),
        ErrorBot,
        GAPredictionBot,
        GeneticAlgorithmBot,
        IPOBot,
        _wrap(ImplementationOptimiserBot, context_builder=builder),
        MirrorBot,
        _wrap(
            NicheSaturationBot,
            alloc_bot=ResourceAllocationBot(context_builder=builder),
        ),
        MarketManipulationBot,
        PassiveDiscoveryBot,
        PreliminaryResearchBot,
        ReportGenerationBot,
        _wrap(ResourceAllocationBot, context_builder=builder),
        _wrap(
            ResourcesBot,
            alloc_bot=ResourceAllocationBot(context_builder=builder),
        ),
        ScalabilityAssessmentBot,
        StrategyPredictionBot,
        StructuralEvolutionBot,
        TextResearchBot,
        VideoResearchBot,
        AICounterBot,
        _wrap(
            DynamicResourceAllocator,
            alloc_bot=ResourceAllocationBot(context_builder=builder),
        ),
        DiagnosticManager,
        KeywordBank,
        NewsDB,
    ]
    for cls in bot_classes:
        try:
            if isinstance(cls, type):
                cls()
            else:
                cls()
        except Exception as exc:
            name = getattr(cls, "__name__", repr(cls))
            logger.exception("Failed to instantiate %s: %s", name, exc)


def run_once(models: Iterable[str]) -> None:
    """Run a single automation cycle for the given models."""
    _init_unused_bots()
    orchestrator = MenaceOrchestrator(context_builder=ContextBuilder())
    # Create a default root oversight bot
    orchestrator.create_oversight("root", "L1")
    override_svc = SelfServiceOverride(ROIDB(), MetricsDB())
    try:
        results = orchestrator.run_cycle(models)
        override_svc.adjust()
    except Exception as exc:
        try:  # best effort ErrorBot logging
            from menace.error_bot import ErrorBot  # type: ignore

            err_bot = ErrorBot(context_builder=ContextBuilder())  # type: ignore[call-arg]
            if hasattr(err_bot, "handle_error"):
                err_bot.handle_error(str(exc))
        except Exception:
            logger.exception("error capture failed")
        logger.exception("run_once failed: %s", exc)
        raise
    for model, res in results.items():
        logger.info(
            "model run result",
            extra=log_record(model=model, result=res),
        )


def _first_run_flag(path: str) -> Path:
    """Return sentinel Path for the first run flag."""
    return Path(path)


def _first_run_completed(path: str) -> bool:
    """Return ``True`` if the sentinel exists."""
    return _first_run_flag(path).exists()


def _mark_first_run(path: str) -> None:
    """Create the sentinel file to indicate first run finished."""
    flag = _first_run_flag(path)
    try:
        flag.touch()
    except Exception:
        logger.exception("failed creating first run flag %s", flag)


def deploy_patch(
    path: Path, description: str, context_builder: "ContextBuilder"
) -> None:
    """Apply a patch using approval policy before deployment."""
    rb = AutomatedRollbackManager()
    policy = PatchApprovalPolicy(rollback_mgr=rb)
    from menace.self_coding_engine import SelfCodingEngine
    from menace.code_database import CodeDB
    from menace.menace_memory_manager import MenaceMemoryManager
    from menace.model_automation_pipeline import ModelAutomationPipeline

    builder = context_builder
    builder.refresh_db_weights()
    engine = SelfCodingEngine(
        CodeDB(), MenaceMemoryManager(), context_builder=builder
    )
    pipeline = ModelAutomationPipeline(context_builder=builder)
    manager = SelfCodingManager(engine, pipeline, approval_policy=policy)
    manager.context_builder = builder
    try:
        manager.run_patch(path, description)
    except Exception:
        rb.auto_rollback("latest", [])


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point running automation in a continuous loop."""
    args = _parse_args(argv)

    _auto_service_setup()

    for item in args.env:
        if "=" in item:
            k, v = item.split("=", 1)
            os.environ[k] = v

    if args.sandbox or os.getenv("MENACE_SANDBOX") == "1":
        sandbox_runner._run_sandbox(args)
        # reload args in case environment changed
        args = _parse_args(argv)

    if args.run_cycles is not None:
        os.environ["RUN_CYCLES"] = str(args.run_cycles)
    if args.run_until is not None:
        os.environ["RUN_UNTIL"] = str(args.run_until)

    if os.getenv("USE_SUPERVISOR") == "1":
        from menace.service_supervisor import main as sup_main

        sup_main()
        return

    os.environ["MENACE_ENV_FILE"] = args.env_file
    ensure_env(args.env_file)
    try:
        interactive_setup(
            defaults_file=args.defaults_file or None,
        )
    except Exception as exc:
        logger.error("setup failed: %s", exc)
    ensure_proxies()
    ensure_accounts()

    first_run_flag = args.first_run_flag
    auto_sandbox_env = os.getenv("AUTO_SANDBOX", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if auto_sandbox_env and not _first_run_completed(first_run_flag):
        try:
            run_once(os.environ.get("MODELS", "demo").split(","))
            sandbox_runner._run_sandbox(args)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("first run failed: %s", exc)
        else:
            _mark_first_run(first_run_flag)
        args = _parse_args(argv)

    _config_store = UnifiedConfigStore(args.env_file)
    _config_store.load()
    _config_store.start_auto_refresh()
    if not args.skip_install:
        try:
            self_check()
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("dependency installation failed: %s", exc)

    _start_dependency_watchdog()
    if os.getenv("AUTO_BOOTSTRAP", "1").lower() not in {"0", "false", "no"}:
        EnvironmentBootstrapper().bootstrap()
    if os.getenv("AUTO_BACKUP") == "1":
        DisasterRecovery(["models.db"]).backup()

    update_stop = threading.Event()
    if os.getenv("AUTO_UPDATE", "1").lower() not in {"0", "false", "no"}:
        from menace.unified_update_service import UnifiedUpdateService

        UnifiedUpdateService().run_continuous(
            interval=float(os.getenv("UPDATE_INTERVAL", "86400")),
            stop_event=update_stop,
        )
    _init_unused_bots()
    from menace.override_policy import OverridePolicyManager, OverrideDB
    from menace.chatgpt_enhancement_bot import EnhancementDB

    policy_mgr = OverridePolicyManager(OverrideDB(), EnhancementDB())
    policy_stop = threading.Event()
    policy_thread = policy_mgr.run_continuous(
        interval=float(os.getenv("OVERRIDE_UPDATE_INTERVAL", "600")),
        stop_event=policy_stop,
    )
    models_env = os.environ.get("MODELS", "demo").split(",")
    sleep_seconds = float(os.environ.get("SLEEP_SECONDS", "0"))
    run_cycles = int(os.environ.get("RUN_CYCLES", "0"))
    _run_until = os.environ.get("RUN_UNTIL")
    run_until = float(_run_until) if _run_until else None
    cycle_count = 0

    stop_event = threading.Event()
    learning_thread = threading.Thread(
        target=learning_service_main, kwargs={"stop_event": stop_event}, daemon=True
    )
    learning_thread.start()

    orchestrator = MenaceOrchestrator(context_builder=ContextBuilder())
    orchestrator.create_oversight("root", "L1")
    orchestrator.start_scheduled_jobs()
    override_svc = SelfServiceOverride(ROIDB(), MetricsDB())
    override_stop = threading.Event()
    override_thread = override_svc.run_continuous(
        interval=60.0,
        stop_event=override_stop,
    )

    try:
        while not stop_event.is_set():
            if run_cycles and cycle_count >= run_cycles:
                break
            if run_until is not None and time.time() >= run_until:
                break
            results = orchestrator.run_cycle(models_env)
            cycle_count += 1
            for model, res in results.items():
                logger.info(
                    "model run result",
                    extra=log_record(model=model, result=res),
                )

            if sleep_seconds < 0:
                break
            if sleep_seconds > 0:
                for _ in range(int(sleep_seconds * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        learning_thread.join()
        update_stop.set()
        policy_stop.set()
        policy_thread.join()
        override_stop.set()
        override_thread.join()


if __name__ == "__main__":
    main()
