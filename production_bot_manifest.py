from __future__ import annotations

"""Canonical manifest of supervisor-manageable services.

A bot is considered "intended for production" when its manifest entry sets
``intended_for_production=True``. This keeps production intent colocated with
startup metadata so registry and validation logic share a single source of
truth.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionBotManifestEntry:
    name: str
    startup_module: str
    startup_callable: str
    health_endpoint: str | None = None
    liveness_check: str = "process_alive"
    critical: bool = False
    enabled_if_env: str | None = None
    needs_context_builder: bool = False
    intended_for_production: bool = True


PRODUCTION_BOT_MANIFEST: tuple[ProductionBotManifestEntry, ...] = (
    ProductionBotManifestEntry(
        name="orchestrator",
        startup_module="service_supervisor",
        startup_callable="_orchestrator_worker",
        needs_context_builder=True,
    ),
    ProductionBotManifestEntry(
        "microtrend_service",
        "service_supervisor",
        "_microtrend_worker",
    ),
    ProductionBotManifestEntry(
        "self_evaluation_service",
        "service_supervisor",
        "_self_eval_worker",
    ),
    ProductionBotManifestEntry(
        "self_learning_service",
        "service_supervisor",
        "_learning_worker",
        critical=True,
    ),
    ProductionBotManifestEntry(
        "model_ranking_service",
        "service_supervisor",
        "_ranking_worker",
    ),
    ProductionBotManifestEntry(
        "dependency_update_service",
        "service_supervisor",
        "_dep_update_worker",
    ),
    ProductionBotManifestEntry(
        name="chaos_monitoring_service",
        startup_module="service_supervisor",
        startup_callable="_chaos_worker",
        needs_context_builder=True,
    ),
    ProductionBotManifestEntry(
        "model_evaluation_service",
        "service_supervisor",
        "_eval_worker",
    ),
    ProductionBotManifestEntry(
        name="debug_loop_service",
        startup_module="service_supervisor",
        startup_callable="_debug_worker",
        needs_context_builder=True,
    ),
    ProductionBotManifestEntry(
        "dependency_watchdog",
        "service_supervisor",
        "_dependency_provision_worker",
    ),
    ProductionBotManifestEntry(
        "dependency_monitor",
        "service_supervisor",
        "_dependency_monitor_worker",
    ),
    ProductionBotManifestEntry(
        "environment_restoration",
        "service_supervisor",
        "_env_restore_worker",
    ),
    ProductionBotManifestEntry(
        "unified_update_service",
        "service_supervisor",
        "_update_worker",
    ),
    ProductionBotManifestEntry(
        name="self_test_service",
        startup_module="service_supervisor",
        startup_callable="_self_test_worker",
        needs_context_builder=True,
    ),
    ProductionBotManifestEntry(
        name="autoscaler",
        startup_module="service_supervisor",
        startup_callable="_autoscale_worker",
        enabled_if_env="ENABLE_AUTOSCALER",
    ),
    ProductionBotManifestEntry(
        name="secret_rotation_service",
        startup_module="service_supervisor",
        startup_callable="_secret_rotation_worker",
        enabled_if_env="AUTO_ROTATE_SECRETS",
    ),
)


__all__ = ["ProductionBotManifestEntry", "PRODUCTION_BOT_MANIFEST"]
