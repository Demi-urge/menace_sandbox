from __future__ import annotations

"""Helper for prompt strategy rotation."""

from pathlib import Path

from ..sandbox_settings import SandboxSettings
from .prompt_strategy_manager import (
    KEYWORD_MAP as BASE_KEYWORD_MAP,
    PromptStrategyManager,
)


KEYWORD_MAP = dict(BASE_KEYWORD_MAP)
KEYWORD_MAP.update({
    "tests_failed": "unit_test_rewrite",
    "comment": "comment_refactor",
})

STATE_PATH = Path(SandboxSettings().sandbox_data_dir) / "strategy_rotator_state.json"
settings = SandboxSettings()
manager = PromptStrategyManager(state_path=STATE_PATH, keyword_map=KEYWORD_MAP)
manager.failure_limits.update(settings.strategy_failure_limits)


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
