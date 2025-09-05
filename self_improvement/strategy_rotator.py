from __future__ import annotations

"""Helper for prompt strategy rotation."""

from .prompt_strategy_manager import PromptStrategyManager

manager = PromptStrategyManager()


def next_strategy(strategy: str | None, failure_reason: str | None = None) -> str | None:
    """Return the next strategy to try after a failure.

    Parameters
    ----------
    strategy:
        The strategy that was attempted.
    failure_reason:
        Optional description of why the attempt failed.
    """

    return manager.record_failure(strategy, failure_reason)


__all__ = ["next_strategy", "manager"]
