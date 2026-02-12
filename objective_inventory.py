"""Canonical objective-surface inventory used by policy and hash-lock checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveInventoryItem:
    """Repository-relative objective surface entry."""

    path: str
    include_in_hash_lock: bool = True


# Canonical objective inventory across reward/evaluator/KPI/ledger/payout/control-plane.
# Keep this list architecture-oriented and repo-relative.
CANONICAL_OBJECTIVE_INVENTORY: tuple[ObjectiveInventoryItem, ...] = (
    ObjectiveInventoryItem("config/objective_hash_lock.json", include_in_hash_lock=False),
    ObjectiveInventoryItem("objective_guard.py"),
    ObjectiveInventoryItem("objective_hash_lock.py"),
    ObjectiveInventoryItem("tools/objective_guard_manifest_cli.py"),
    ObjectiveInventoryItem("reward_dispatcher.py"),
    ObjectiveInventoryItem("kpi_reward_core.py"),
    ObjectiveInventoryItem("reward_sanity_checker.py"),
    ObjectiveInventoryItem("kpi_editing_detector.py"),
    ObjectiveInventoryItem("mvp_evaluator.py"),
    ObjectiveInventoryItem("evaluation_worker.py"),
    ObjectiveInventoryItem("evaluation_manager.py"),
    ObjectiveInventoryItem("evaluation_service.py"),
    ObjectiveInventoryItem("self_evaluation_service.py"),
    ObjectiveInventoryItem("model_evaluation_service.py"),
    ObjectiveInventoryItem("evaluation_history_db.py"),
    ObjectiveInventoryItem("central_evaluation_loop.py"),
    ObjectiveInventoryItem("menace/core/evaluator.py"),
    ObjectiveInventoryItem("neurosales/neurosales/hierarchical_reward.py"),
    ObjectiveInventoryItem("neurosales/neurosales/reward_ledger.py"),
    ObjectiveInventoryItem("billing/billing_ledger.py"),
    ObjectiveInventoryItem("billing/billing_logger.py"),
    ObjectiveInventoryItem("billing/stripe_ledger.py"),
    ObjectiveInventoryItem("stripe_billing_router.py"),
    ObjectiveInventoryItem("finance_router_bot.py"),
    ObjectiveInventoryItem("stripe_watchdog.py"),
    ObjectiveInventoryItem("startup_health_check.py"),
    ObjectiveInventoryItem("finance_logs/", include_in_hash_lock=False),
)


OBJECTIVE_ADJACENT_UNSAFE_PATHS: tuple[str, ...] = tuple(
    item.path for item in CANONICAL_OBJECTIVE_INVENTORY
)
OBJECTIVE_ADJACENT_HASH_PATHS: tuple[str, ...] = tuple(
    item.path for item in CANONICAL_OBJECTIVE_INVENTORY if item.include_in_hash_lock
)


__all__ = [
    "CANONICAL_OBJECTIVE_INVENTORY",
    "OBJECTIVE_ADJACENT_HASH_PATHS",
    "OBJECTIVE_ADJACENT_UNSAFE_PATHS",
    "ObjectiveInventoryItem",
]
