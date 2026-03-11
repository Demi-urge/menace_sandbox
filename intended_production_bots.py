from __future__ import annotations

"""Canonical list of supervisor-managed production services.

The names in :data:`INTENDED_PRODUCTION_BOTS` represent the services that should
exist in ``RUNNABLE_BOT_REGISTRY`` so they are eligible for supervisor launch
and runtime health policy enforcement.
"""

INTENDED_PRODUCTION_BOTS: tuple[str, ...] = (
    "orchestrator",
    "microtrend_service",
    "self_evaluation_service",
    "self_learning_service",
    "model_ranking_service",
    "dependency_update_service",
    "chaos_monitoring_service",
    "model_evaluation_service",
    "debug_loop_service",
    "dependency_watchdog",
    "dependency_monitor",
    "environment_restoration",
    "unified_update_service",
    "self_test_service",
    "autoscaler",
    "secret_rotation_service",
)


__all__ = ["INTENDED_PRODUCTION_BOTS"]
