"""Menace package initialization logic.

This module also exposes :class:`ROICalculator` whose profiles live in
``configs/roi_profiles.yaml``.
"""

import importlib.util
import logging
import os
import sys
import shutil
import time
from typing import TYPE_CHECKING
import types
from pathlib import Path

# Allow optional exception propagation in modules that normally swallow errors.
# The flag needs to be defined before importing any submodules because several of
# them import :mod:`config_discovery` during their module initialisation.  That
# module in turn performs ``from menace_sandbox import RAISE_ERRORS``.  When the
# attribute was defined at the bottom of this file the circular import meant the
# name was missing which surfaced as ``ImportError: cannot import name
# 'RAISE_ERRORS'`` when running the docker entrypoint.  Defining it eagerly keeps
# the attribute available throughout module import, avoiding the circular
# dependency.
RAISE_ERRORS = os.getenv("MENACE_RAISE_ERRORS") == "1"

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def _resolve_with_package_root(target: os.PathLike[str] | str) -> Path:
    """Return *target* as a :class:`Path` rooted at the package directory."""

    path = Path(target)
    if path.is_absolute():
        return path
    return (_PACKAGE_ROOT / path).resolve(strict=False)


def _resolve_module_with_package_root(module: str) -> Path:
    """Return the filesystem path for ``module`` relative to the package."""

    module_path = Path(module.replace(".", "/"))
    return _resolve_with_package_root(module_path.with_suffix(".py"))


def _path_for_prompt_stub(target: os.PathLike[str] | str) -> str:
    """Return a POSIX path suitable for prompt ingestion."""

    return _resolve_with_package_root(target).as_posix()


sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(
        resolve_path=_resolve_with_package_root,
        resolve_dir=_resolve_with_package_root,
        resolve_module_path=_resolve_module_with_package_root,
        path_for_prompt=_path_for_prompt_stub,
        get_project_root=lambda: _PACKAGE_ROOT,
    ),
)

# Provide a legacy alias expected by some bootstrapping utilities.
sys.modules.setdefault("menace", sys.modules[__name__])

from .roi_calculator import ROICalculator
# ErrorParser is optional during lightweight imports; fall back to None if heavy
# dependencies are missing.
try:  # pragma: no cover - best effort import
    from .error_parser import ErrorParser
except Exception:  # pragma: no cover - gracefully degrade in tests
    ErrorParser = None  # type: ignore
from .truth_adapter import TruthAdapter
from .foresight_tracker import ForesightTracker
from .upgrade_forecaster import UpgradeForecaster
try:  # pragma: no cover - optional heavy dependency
    from .workflow_synthesizer import WorkflowSynthesizer
    from .workflow_synergy_comparator import WorkflowSynergyComparator
except Exception:  # pragma: no cover - degrade gracefully for tests
    WorkflowSynthesizer = None  # type: ignore
    WorkflowSynergyComparator = None  # type: ignore
from .llm_interface import LLMClient, LLMResult, Prompt

from . import metrics_exporter

_ORIG_RMTREE = shutil.rmtree

_rmtree_retries = metrics_exporter.Gauge(
    "cleanup_rmtree_retries_total",
    "Total retries while removing directories on Windows",
    labelnames=("path",),
)

_rmtree_failures = metrics_exporter.Gauge(
    "cleanup_rmtree_failures_total",
    "Total failed directory removals on Windows",
    labelnames=("path",),
)


def _rmtree_with_retry(
    path: str,
    *args: object,
    attempts: int = 3,
    delay: float = 0.1,
    **kwargs: object,
) -> None:
    """Remove ``path`` with retries and metrics on Windows."""

    if os.name != "nt":
        return _ORIG_RMTREE(path, *args, **kwargs)

    ignore_errors = bool(kwargs.get("ignore_errors", False))
    for i in range(attempts):
        try:
            return _ORIG_RMTREE(path, *args, **kwargs)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            if i < attempts - 1:
                logging.warning(
                    "rmtree attempt %s/%s for %s failed: %s",
                    i + 1,
                    attempts,
                    path,
                    exc,
                )
                try:
                    _rmtree_retries.labels(str(path)).inc()
                except Exception:
                    pass
                time.sleep(delay * (2 ** i))
            else:
                logging.error(
                    "rmtree failed for %s after %s attempts: %s",
                    path,
                    attempts,
                    exc,
                )
                try:
                    _rmtree_failures.labels(str(path)).inc()
                except Exception:
                    pass
                if not ignore_errors:
                    raise
                return


shutil.rmtree = _rmtree_with_retry
if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .composite_workflow_scorer import CompositeWorkflowScorer
try:  # pragma: no cover - optional heavy dependency
    from .intent_clusterer import IntentClusterer
except Exception:  # pragma: no cover - gracefully degrade
    IntentClusterer = None  # type: ignore[misc]


def __getattr__(name: str) -> object:
    """Lazily import optional heavy modules on first access.

    This keeps initial import costs low while still exposing the class via
    ``menace_sandbox.CompositeWorkflowScorer``.  If the import fails the
    attribute resolves to ``None`` instead of raising ``ImportError``.
    """

    if name == "CompositeWorkflowScorer":  # pragma: no cover - dynamic import
        try:
            from .composite_workflow_scorer import (
                CompositeWorkflowScorer as _CompositeWorkflowScorer,
            )
        except Exception:  # pragma: no cover - gracefully degrade
            globals()[name] = None
            return None
        globals()[name] = _CompositeWorkflowScorer
        return _CompositeWorkflowScorer
    raise AttributeError(name)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# If production mode is used with the default SQLite database, fall back to test
if os.getenv("MENACE_MODE", "test").lower() == "production" and os.getenv(
    "DATABASE_URL", ""
).startswith("sqlite"):
    logging.warning(
        "MENACE_MODE=production with SQLite database; switching to test mode"
    )
    os.environ["MENACE_MODE"] = "test"

if not os.getenv("MENACE_LIGHT_IMPORTS"):
    from . import menace_db as _menace_db

    sys.modules.setdefault("menace.menace", _menace_db)
else:
    import types

    _menace_db = types.ModuleType("menace.menace")
    sys.modules.setdefault("menace.menace", _menace_db)

import importlib

_log_utils = importlib.import_module(__name__ + ".logging_utils")
sys.modules.setdefault("logging_utils", _log_utils)
sys.modules.setdefault("menace.logging_utils", _log_utils)
_alert_dispatcher = importlib.import_module(__name__ + ".alert_dispatcher")
sys.modules.setdefault("alert_dispatcher", _alert_dispatcher)
sys.modules.setdefault("menace.alert_dispatcher", _alert_dispatcher)
_readiness_index = importlib.import_module(__name__ + ".readiness_index")
sys.modules.setdefault("readiness_index", _readiness_index)
sys.modules.setdefault("menace.readiness_index", _readiness_index)


class _LegacyModule(types.ModuleType):
    """Lazy proxy that exposes ``menace_sandbox.gpt_memory`` as ``gpt_memory``."""

    _loaded: types.ModuleType | None

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._loaded = None

    def _load(self) -> types.ModuleType:
        if self._loaded is None:
            module = importlib.import_module(__name__ + ".gpt_memory")
            self._loaded = module
            sys.modules[self.__name__] = module
            return module
        return self._loaded

    def __getattr__(self, item: str) -> object:
        return getattr(self._load(), item)

    def __dir__(self) -> list[str]:
        return dir(self._load())


sys.modules.setdefault("gpt_memory", _LegacyModule("gpt_memory"))
from .dynamic_path_router import resolve_path, resolve_module_path, resolve_dir, get_project_root

_sk_dir = get_project_root() / "sklearn"
_extra_dir = get_project_root() / "menace"
if _extra_dir.is_dir() and str(_extra_dir) not in __path__:
    __path__.append(str(_extra_dir))
if "sklearn" not in sys.modules and _sk_dir.is_dir():
    spec = importlib.util.spec_from_file_location(
        "sklearn", str(resolve_path("sklearn/__init__.py"))
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["sklearn"] = mod
    for sub in ["feature_extraction", "linear_model", "pipeline"]:
        spec_sub = importlib.util.spec_from_file_location(
            f"sklearn.{sub}", str(resolve_path(f"sklearn/{sub}/__init__.py"))
        )
        mod_sub = importlib.util.module_from_spec(spec_sub)
        spec_sub.loader.exec_module(mod_sub)
        sys.modules[f"sklearn.{sub}"] = mod_sub

logger = logging.getLogger(__name__)

# Allow optional exception propagation in modules that normally swallow errors.
# (See comment near top of file for reasoning.)

if not os.getenv("MENACE_LIGHT_IMPORTS"):
    _submodules = [
        "allocator_service",
        "databases",
        "identity_seeder",
        "proxy_broker",
        "session_aware_http",
        "session_vault",
    ]

    _failed_imports: list[str] = []
    for _name in _submodules:
        try:
            globals()[_name] = importlib.import_module(f"menace.{_name}")
        except Exception as exc:  # pragma: no cover - optional modules may be missing
            logger.exception("Failed to import submodule %s: %s", _name, exc)
            _failed_imports.append(_name)

    if _failed_imports:  # pragma: no cover - informational
        logger.warning(
            "Menace submodules failed to import: %s", ", ".join(_failed_imports)
        )

__all__ = [
    "RAISE_ERRORS",
    "__version__",
    "ROICalculator",
    "roi_calculator",
    "ErrorParser",
    "error_parser",
    "TruthAdapter",
    "truth_adapter",
    "ForesightTracker",
    "foresight_tracker",
    "UpgradeForecaster",
    "upgrade_forecaster",
    "WorkflowSynthesizer",
    "workflow_synthesizer",
    "WorkflowSynergyComparator",
    "IntentClusterer",
    "intent_clusterer",
    "newsreader_bot",
    "preliminary_research_bot",
    "passive_discovery_bot",
    "research_aggregator_bot",
    "chatgpt_research_bot",
    "chatgpt_enhancement_bot",
    "chatgpt_idea_bot",
    "chatgpt_prediction_bot",
    "contrarian_model_bot",
    "information_synthesis_bot",
    "task_validation_bot",
    "bot_planning_bot",
    "hierarchy_assessment_bot",
    "resource_prediction_bot",
    "model_automation_pipeline",
    "text_research_bot",
    "video_research_bot",
    "task_handoff_bot",
    "genetic_algorithm_bot",
    "ga_prediction_bot",
    "prediction_manager_bot",
    "pre_execution_roi_bot",
    "bot_development_bot",
    "scalability_assessment_bot",
    "bot_testing_bot",
    "resource_allocation_bot",
    "ipo_bot",
    "data_bot",
    "metrics_aggregator",
    "deployment_bot",
    "bot_database",
    "code_database",
    "information_db",
    "error_bot",
    "error_forecaster",
    "error_cluster_predictor",
    "ipo_implementation_pipeline",
    "database_steward_bot",
    "database_management_bot",
    "bot_creation_bot",
    "admin_bot_base",
    "competitive_intelligence_bot",
    "strategy_prediction_bot",
    "ai_counter_bot",
    "sentiment_bot",
    "resources_bot",
    "niche_saturation_bot",
    "efficiency_bot",
    "energy_forecast_bot",
    "future_prediction_bots",
    "report_generation_bot",
    "conversation_manager_bot",
    "mirror_bot",
    "performance_assessment_bot",
    "bot_performance_history_db",
    "memory_bot",
    "query_bot",
    "communication_testing_bot",
    "structural_evolution_bot",
    "dynamic_resource_allocator_bot",
    "coordination_manager",
    "menace_memory_manager",
    "ga_clone_manager",
    "system_evolution_manager",
    "evolution_orchestrator",
    "evolution_history_db",
    "experiment_manager",
    "experiment_history_db",
    "evolution_analysis_bot",
    "evolution_scheduler",
    "menace_orchestrator",
    "meta_genetic_algorithm_bot",
    "personalized_conversation",
    "user_style_model",
    "market_manipulation_bot",
    "diagnostic_manager",
    "capital_management_bot",
    "operational_monitor_bot",
    "communication_maintenance_bot",
    "finance_router_bot",
    "menace_gui",
    "revenue_amplifier",
    "menace_db",
    "normalize_scraped_data",
    "trending_scraper",
    "candidate_matcher",
    "offer_testing_bot",
    "failure_learning_system",
    "investment_engine",
    "contrarian_db",
    "prime_utils",
    "implementation_optimiser_bot",
    "scalability_pipeline",
    "implementation_pipeline",
    "prediction_training_pipeline",
    "neuroplasticity",
    "research_fallback_bot",
    "resource_allocation_optimizer",
    "contextual_rl",
    "trial_monitor",
    "discrepancy_detection_bot",
    "override_policy",
    "override_db",
    "stats",
    "meta_logging",
    "replay_training",
    "central_database_bot",
    "auto_archiver_bot",
    "anticaptcha_stub",
    "captcha_system",
    "replay_engine",
    "proxy_broker",
    "whisper_utils",
    "oversight_bots",
    "self_improvement",
    "self_model_bootstrap",
    "unified_event_bus",
    "networked_event_bus",
    "neo4j_listener",
    "bot_registry",
    "event_collector",
    "capability_registry",
    "self_learning_coordinator",
    "self_learning_service",
    "learning_engine",
    "preprocessing_utils",
    "chaos_scheduler",
    "advanced_error_management",
    "automated_debugger",
    "microtrend_service",
    "service_supervisor",
    "model_deployer",
    "cross_model_comparator",
    "cross_model_scheduler",
    "self_validation_dashboard",
    "sandbox_dashboard",
    "auto_escalation_manager",
    "self_service_override",
    "infrastructure_bootstrap",
    "self_evaluation_service",
    "unified_update_service",
    "debug_loop_service",
    "cluster_supervisor",
    "compliance_checker",
    "strategic_planner",
    "vault_secret_provider",
    "environment_restoration_service",
    "self_test_service",
    "default_config_manager",
    "auto_env_setup",
    "auto_resource_setup",
    "external_dependency_provisioner",
    "disaster_recovery",
    "supervisor_watchdog",
    "indirect_domain_reference_detector",
    "evolution_ab_comparator",
    "mutation_logger",
    "module_index_db",
    "synergy_history_db",
    "synergy_auto_trainer",
    "variant_manager",
    "cognition_layer",
    "CompositeWorkflowScorer",
    "Prompt",
    "LLMResult",
    "LLMClient",
    "llm_interface",
]
__all__.append("readiness_index")
__version__ = "0.1.0"
