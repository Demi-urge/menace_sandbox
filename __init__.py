"""Menace package initialization logic.

This module also exposes :class:`ROICalculator` whose profiles live in
``configs/roi_profiles.yaml``.
"""

import importlib.util
import logging
import os
import sys

from .roi_calculator import ROICalculator
from .truth_adapter import TruthAdapter
from .foresight_tracker import ForesightTracker
from .upgrade_forecaster import UpgradeForecaster
from .workflow_synthesizer import WorkflowSynthesizer
CompositeWorkflowScorer = None  # type: ignore
try:  # pragma: no cover - optional heavy dependency
    from .intent_clusterer import IntentClusterer
except Exception:  # pragma: no cover - gracefully degrade
    IntentClusterer = None  # type: ignore[misc]

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

_pkg_dir = os.path.dirname(__file__)
_sk_dir = os.path.join(_pkg_dir, "sklearn")
_extra_dir = os.path.join(_pkg_dir, "menace")
if os.path.isdir(_extra_dir) and _extra_dir not in __path__:
    __path__.append(_extra_dir)
if "sklearn" not in sys.modules and os.path.isdir(_sk_dir):
    spec = importlib.util.spec_from_file_location(
        "sklearn", os.path.join(_sk_dir, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["sklearn"] = mod
    for sub in ["feature_extraction", "linear_model", "pipeline"]:
        spec_sub = importlib.util.spec_from_file_location(
            f"sklearn.{sub}", os.path.join(_sk_dir, sub, "__init__.py")
        )
        mod_sub = importlib.util.module_from_spec(spec_sub)
        spec_sub.loader.exec_module(mod_sub)
        sys.modules[f"sklearn.{sub}"] = mod_sub

logger = logging.getLogger(__name__)

# Allow optional exception propagation in modules that normally swallow errors.
RAISE_ERRORS = os.getenv("MENACE_RAISE_ERRORS") == "1"

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
    "__version__",
    "ROICalculator",
    "roi_calculator",
    "TruthAdapter",
    "truth_adapter",
    "ForesightTracker",
    "foresight_tracker",
    "UpgradeForecaster",
    "upgrade_forecaster",
    "WorkflowSynthesizer",
    "workflow_synthesizer",
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
    "vision_utils",
    "oversight_bots",
    "self_improvement_engine",
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
]
__all__.append("readiness_index")
__version__ = "0.1.0"
