"""Single source of truth for objective-adjacent mutation deny rules."""

from __future__ import annotations

# NOTE: Keep these paths repo-relative and architecture-oriented.
# They describe the objective surface (rewarding/evaluation/payout/ledger logic)
# that self-coding flows must not mutate automatically.
#
# Canonical model:
# - ``OBJECTIVE_ADJACENT_UNSAFE_PATHS`` is the canonical inventory for
#   pre-mutation blocking/manual approval.
# - ``OBJECTIVE_ADJACENT_HASH_PATHS`` is the canonical inventory for hash-lock
#   coverage. It intentionally excludes ``config/objective_hash_lock.json`` to
#   avoid self-referential hash drift during manifest writes.
OBJECTIVE_ADJACENT_UNSAFE_PATHS: tuple[str, ...] = (
    # Objective control-plane artifacts (hash lock + guard tooling).
    "config/objective_hash_lock.json",
    "objective_guard.py",
    "objective_hash_lock.py",
    "tools/objective_guard_manifest_cli.py",
    # Reward/evaluator/KPI mapping surface.
    "reward_dispatcher.py",
    "kpi_reward_core.py",
    "reward_sanity_checker.py",
    "kpi_editing_detector.py",
    "mvp_evaluator.py",
    "evaluation_worker.py",
    "evaluation_manager.py",
    "evaluation_service.py",
    "self_evaluation_service.py",
    "model_evaluation_service.py",
    "evaluation_history_db.py",
    "central_evaluation_loop.py",
    "menace/core/evaluator.py",
    "neurosales/neurosales/hierarchical_reward.py",
    "neurosales/neurosales/reward_ledger.py",
    # Ledger + payout routing surface.
    "billing/billing_ledger.py",
    "billing/billing_logger.py",
    "billing/stripe_ledger.py",
    "stripe_billing_router.py",
    "finance_router_bot.py",
    "stripe_watchdog.py",
    "startup_health_check.py",
    "finance_logs/",
)

OBJECTIVE_ADJACENT_HASH_PATHS: tuple[str, ...] = (
    "objective_guard.py",
    "objective_hash_lock.py",
    "tools/objective_guard_manifest_cli.py",
    "reward_dispatcher.py",
    "kpi_reward_core.py",
    "reward_sanity_checker.py",
    "kpi_editing_detector.py",
    "mvp_evaluator.py",
    "evaluation_worker.py",
    "evaluation_manager.py",
    "evaluation_service.py",
    "self_evaluation_service.py",
    "model_evaluation_service.py",
    "evaluation_history_db.py",
    "central_evaluation_loop.py",
    "menace/core/evaluator.py",
    "neurosales/neurosales/hierarchical_reward.py",
    "neurosales/neurosales/reward_ledger.py",
    "billing/billing_ledger.py",
    "billing/billing_logger.py",
    "billing/stripe_ledger.py",
    "stripe_billing_router.py",
    "finance_router_bot.py",
    "stripe_watchdog.py",
    "startup_health_check.py",
)

__all__ = ["OBJECTIVE_ADJACENT_HASH_PATHS", "OBJECTIVE_ADJACENT_UNSAFE_PATHS"]
