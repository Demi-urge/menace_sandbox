from __future__ import annotations

"""Central registry for supervisor-managed runnable services."""

from dataclasses import dataclass

from intended_production_bots import INTENDED_PRODUCTION_BOTS


@dataclass(frozen=True)
class RunnableBotEntry:
    name: str
    startup_module: str
    startup_callable: str
    health_endpoint: str | None = None
    liveness_check: str = "process_alive"
    critical: bool = False
    enabled_if_env: str | None = None
    needs_context_builder: bool = False


RUNNABLE_BOT_REGISTRY: tuple[RunnableBotEntry, ...] = (
    RunnableBotEntry(
        name="orchestrator",
        startup_module="service_supervisor",
        startup_callable="_orchestrator_worker",
        needs_context_builder=True,
    ),
    RunnableBotEntry("microtrend_service", "service_supervisor", "_microtrend_worker"),
    RunnableBotEntry("self_evaluation_service", "service_supervisor", "_self_eval_worker"),
    RunnableBotEntry(
        "self_learning_service",
        "service_supervisor",
        "_learning_worker",
        critical=True,
    ),
    RunnableBotEntry("model_ranking_service", "service_supervisor", "_ranking_worker"),
    RunnableBotEntry("dependency_update_service", "service_supervisor", "_dep_update_worker"),
    RunnableBotEntry(
        name="chaos_monitoring_service",
        startup_module="service_supervisor",
        startup_callable="_chaos_worker",
        needs_context_builder=True,
    ),
    RunnableBotEntry("model_evaluation_service", "service_supervisor", "_eval_worker"),
    RunnableBotEntry(
        name="debug_loop_service",
        startup_module="service_supervisor",
        startup_callable="_debug_worker",
        needs_context_builder=True,
    ),
    RunnableBotEntry("dependency_watchdog", "service_supervisor", "_dependency_provision_worker"),
    RunnableBotEntry("dependency_monitor", "service_supervisor", "_dependency_monitor_worker"),
    RunnableBotEntry("environment_restoration", "service_supervisor", "_env_restore_worker"),
    RunnableBotEntry("unified_update_service", "service_supervisor", "_update_worker"),
    RunnableBotEntry(
        name="self_test_service",
        startup_module="service_supervisor",
        startup_callable="_self_test_worker",
        critical=True,
        needs_context_builder=True,
    ),
    RunnableBotEntry(
        name="autoscaler",
        startup_module="service_supervisor",
        startup_callable="_autoscale_worker",
        enabled_if_env="ENABLE_AUTOSCALER",
    ),
    RunnableBotEntry(
        name="secret_rotation_service",
        startup_module="service_supervisor",
        startup_callable="_secret_rotation_worker",
        enabled_if_env="AUTO_ROTATE_SECRETS",
    ),
)


__all__ = ["RunnableBotEntry", "RUNNABLE_BOT_REGISTRY", "INTENDED_PRODUCTION_BOTS"]
