"""Canonical objective-adjacent paths that self-coding patch promotion must deny."""

from __future__ import annotations

OBJECTIVE_ADJACENT_UNSAFE_PATHS: tuple[str, ...] = (
    "kpi_reward_core.py",
    "reward_dispatcher.py",
    "reward_sanity_checker.py",
    "mvp_evaluator.py",
    "kpi_editing_detector.py",
    "menace/core/evaluator.py",
    "billing/billing_ledger.py",
    "billing/billing_logger.py",
    "billing/stripe_ledger.py",
    "finance_router_bot.py",
)

__all__ = ["OBJECTIVE_ADJACENT_UNSAFE_PATHS"]
