"""Single source of truth for objective-adjacent mutation deny rules."""

from __future__ import annotations

# NOTE: Keep these paths repo-relative and architecture-oriented.
# They describe the objective surface (rewarding/evaluation/payout/ledger logic)
# that self-coding flows must not mutate automatically.
OBJECTIVE_ADJACENT_UNSAFE_PATHS: tuple[str, ...] = (
    "reward_dispatcher.py",
    "kpi_reward_core.py",
    "reward_sanity_checker.py",
    "kpi_editing_detector.py",
    "mvp_evaluator.py",
    "menace/core/evaluator.py",
    "billing/billing_ledger.py",
    "billing/billing_logger.py",
    "billing/stripe_ledger.py",
    "stripe_billing_router.py",
    "finance_router_bot.py",
)

__all__ = ["OBJECTIVE_ADJACENT_UNSAFE_PATHS"]

