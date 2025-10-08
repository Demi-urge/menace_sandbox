from __future__ import annotations

"""Helper for prompt strategy rotation."""

from pathlib import Path

from dynamic_path_router import resolve_path
from menace_sandbox.sandbox_settings import SandboxSettings
from .prompt_strategy_manager import (
    KEYWORD_MAP as BASE_KEYWORD_MAP,
    PromptStrategyManager,
)


KEYWORD_MAP = dict(BASE_KEYWORD_MAP)
KEYWORD_MAP.update({
    "tests_failed": "unit_test_rewrite",
    "comment": "comment_refactor",
})

TEMPLATES = [
    "strict_fix",
    "delete_rebuild",
    "comment_refactor",
    "unit_test_rewrite",
]

STATE_PATH = Path(
    resolve_path(
        Path(SandboxSettings().sandbox_data_dir) / "strategy_rotator_state.json"
    )
)
settings = SandboxSettings()
manager = PromptStrategyManager(
    strategies=TEMPLATES, state_path=STATE_PATH, keyword_map=KEYWORD_MAP
)
manager.failure_limits.update(settings.strategy_failure_limits)


def next_strategy(
    strategy: str | None,
    failure_reason: str | None = None,
    roi_delta: float | None = None,
) -> str | None:
    """Return the next strategy to try after a failure.

    Parameters
    ----------
    strategy:
        The strategy that was attempted.
    failure_reason:
        Optional description of why the attempt failed.
    roi_delta:
        ROI change produced by the attempt.
    """

    forced = manager.ingest(strategy, failure_reason, roi_delta)
    if forced:
        return forced
    return manager.next()


__all__ = ["next_strategy", "manager", "TEMPLATES"]
